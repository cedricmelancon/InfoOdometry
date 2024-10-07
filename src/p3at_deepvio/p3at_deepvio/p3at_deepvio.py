import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu

import cv2
from PIL import Image as Img
from threading import Lock
import numpy as np
import torch
import collections

from info_odometry.odometry_model import OdometryModel
from info_odometry.param import Param
from info_odometry.utils.tools import get_absolute_pose_step

from BlazeAIoT.Core.NodeManager import ServiceManager
import time
import csv


class P3atDeepvio(Node):
    def __init__(self):
        super().__init__('P3atDeepvio')
        self._service_manager = ServiceManager()

        self._imu_lock = Lock()
        self._camera_lock = Lock()
        self._model_lock = Lock()

        self.skip_frame = 5
        self.frame_nb = 0
        self.timing_monitor = {}

        param = Param()
        args = param.get_args()
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        publisher_group = ReentrantCallbackGroup()
        subscriber1_group = ReentrantCallbackGroup()
        subscriber2_group = ReentrantCallbackGroup()

        self._odometry_model = OdometryModel(args)

        self._publisher = self.create_publisher(Odometry, '', 10, callback_group=publisher_group)
        self._imu_subscriber = self.create_subscription(Imu, '/torso_lift_imu/data', self.imu_callback, 10, callback_group=subscriber1_group)
        self._camera_subscriber = self.create_subscription(Image, '/camera/rgb/image_raw', self.camera_callback, 10, callback_group=subscriber2_group)
        self._imu_data = collections.deque(maxlen=3)
        self._last_position = [29.271869156149368, 129.52834578074683, 0.0, 0.0, 0.0, 0.34944565567934077]
        self._last_camera_data = None
        self._last_stamp = None
        
        self._img_seq = torch.zeros([self._odometry_model.clip_length, 1, 1024, 8, 10]).to('cuda:0')
        self._imu_seq = collections.deque(maxlen=self._odometry_model.clip_length)
        self._monitoring_data = collections.deque()
        self._odometry_state = torch.zeros(1,
                                           self._odometry_model.args.state_size,
                                           device=self._odometry_model.args.device)
        self._beliefs = None
        self._monitoring_task = None

        self.get_logger().info('Running')

    def write_timing(self):
        csvfile = open(f'monitoring.csv', 'w', newline='')
        self.csvwriter = csv.writer(csvfile, delimiter=' ')

        self.get_logger().info('Writing timing.')

        while len(self._monitoring_data) > 0:
            self.csvwriter.writerow(self._monitoring_data.pop())

        self.get_logger().info('Timing written.')
                

    @staticmethod
    def image_to_tensor(image, width, height):
        image = image.reshape(height, width)
        image = cv2.cvtColor(image, cv2.COLOR_BAYER_GR2BGR)
    
        return image.transpose(1, 0, 2)

    @staticmethod
    def push_to_tensor_alternative(tensor, x):
        return torch.cat((tensor[1:7], x))

    def process_data(self, camera_data, last_camera_data, height, width, current_stamp):
        camera_data = self.image_to_tensor(camera_data, height, width)

        if last_camera_data is not None:
            start_time = time.perf_counter()
            last_camera_data = self.image_to_tensor(last_camera_data, height, width)

            img_pair = [last_camera_data, camera_data]
            img_pair = np.array(img_pair).transpose(3, 0, 1, 2)
            img_pair = np.expand_dims(img_pair, axis=0)
            img_pair = np.expand_dims(img_pair, axis=0)
            img_pair = torch.from_numpy(img_pair).type(torch.FloatTensor).to('cuda:0')

            feature_data = self._odometry_model.eval_flownet_model(img_pair)
            flownet_time = (time.perf_counter() - start_time)

            self.push_to_tensor_alternative(self._img_seq, feature_data)

            self._model_lock.acquire()

            odometry, timing_monitor = self._odometry_model.step_model(
                self._img_seq,
                torch.from_numpy(np.array(list(self._imu_seq))).type(torch.FloatTensor).to('cuda:0'),
                self._odometry_state)
            self._model_lock.release()

            if odometry is not None:
                odometry = odometry.cpu().numpy()[0]
                dt = [float(odometry[0]), float(odometry[1]), 0.0, 0.0, 0.0, float(odometry[5])]

                self._last_position = get_absolute_pose_step(dt, self._last_position)
                odometry_msg = Odometry()
                odometry_msg.header.stamp = current_stamp
                odometry_msg.pose.pose.position.x = float(self._last_position[0])
                odometry_msg.pose.pose.position.y = float(self._last_position[1])
                odometry_msg.pose.pose.position.z = 0.0
                odometry_msg.pose.pose.orientation.x = float(self._last_position[3])
                odometry_msg.pose.pose.orientation.y = float(self._last_position[4])
                odometry_msg.pose.pose.orientation.z = float(self._last_position[5])
                odometry_msg.pose.pose.orientation.w = float(self._last_position[6])

                #if self._last_position is None:
                #    odometry_msg.twist.linear.x = 0.0
                #    odometry_msg.twist.linear.y = 0.0
                #    odometry_msg.twist.linear.z = 0.0

                #    odometry_msg.twist.angular.x = 0.0
                #    odometry_msg.twist.angular.y = 0.0
                #    odometry_msg.twist.angular.z = 0.0
                #else:
                #    odometry_msg.twist.linear.x = 0.0
                #    odometry_msg.twist.linear.y = 0.0
                #    odometry_msg.twist.linear.z = 0.0

                #    odometry_msg.twist.angular.x = 0.0
                #    odometry_msg.twist.angular.y = 0.0
                #    odometry_msg.twist.angular.z = 0.0

                #self.rate_counter.inc()
                self._publisher.publish(odometry_msg)

                #odometry, timing_monitor = self.execute_model(current_stamp)
                
                self._monitoring_data.append(np.array([self.frame_nb, flownet_time] + timing_monitor + [time.perf_counter() - start_time]))


    def camera_callback(self, msg):
        self._imu_lock.acquire()
        imu_data = np.array(list(self._imu_data))
        self._imu_lock.release()

        self._imu_seq.append(np.expand_dims(imu_data, 0))
        if self.frame_nb < self.skip_frame:
            self.frame_nb += 1
            return

        self.frame_nb += 1
        self._camera_lock.acquire()
        camera_data = np.array(list(msg.data), dtype=np.uint8)
        self._camera_lock.release()
        last_camera_data = np.array(self._last_camera_data) if self._last_camera_data is not None else None
        self.process_data(camera_data, 
                          last_camera_data, 
                          msg.height, 
                          msg.width, 
                          msg.header.stamp)

        self._last_camera_data = np.array(camera_data)
        
    def imu_callback(self, msg):
        self._imu_lock.acquire()
        imu_data = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
             msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self._imu_data.append(imu_data)
        self._imu_lock.release()

def main(args=None):
    rclpy.init(args=args)
    executor = rclpy.executors.MultiThreadedExecutor()
    localization_node = P3atDeepvio()
    executor.add_node(localization_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        localization_node.get_logger().info('Shutting down...')
    finally:
        localization_node.write_timing()
        #localization_node.destroy_node()
        #rclpy.shutdown()

if __name__ == '__main__':
    main()
