import rclpy
from rclpy.node import Node

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

import time


class P3atDeepvio(Node):
    def __init__(self):
        super().__init__('P3atDeepvio')
        
        self.background_tasks = set()

        self._imu_lock = Lock()
        self._camera_lock = Lock()
        self._model_lock = Lock()

        self.skip_frame = 5
        self.frame_nb = 0
        param = Param()
        args = param.get_args()
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        self._odometry_model = OdometryModel(args)
        self._publisher = self.create_publisher(Odometry, 'deepvio_odometry', 10)
        self._imu_subscriber = self.create_subscription(Imu, '/torso_lift_imu/data', self.imu_callback, 10)
        self._camera_subscriber = self.create_subscription(Image, '/camera/rgb/image_raw', self.camera_callback, 10)
        self._imu_data = collections.deque(maxlen=3)
        self._imu_seq = collections.deque(maxlen=self._odometry_model.clip_length)
        self._camera_features = collections.deque(maxlen=self._odometry_model.clip_length)
        self._last_position = None
        self._last_camera_data = None
        self._last_stamp = None
        self.get_logger().info('C\'est parti!')
        self._odometry_state = torch.zeros(1,
                                           self._odometry_model.args.state_size,
                                           device=self._odometry_model.args.device)
        self._beliefs = None

    @staticmethod
    def image_to_tensor(image, width, height):
        image = image.reshape(height, width)
        image = cv2.cvtColor(image, cv2.COLOR_BAYER_GR2BGR)
        return image.transpose(1, 0, 2)

    def process_data(self, camera_data, last_camera_data, height, width, imu_seq, current_stamp):
        camera_data = self.image_to_tensor(camera_data, height, width)
        
        if last_camera_data is not None:
            last_camera_data = self.image_to_tensor(last_camera_data, height, width)

            img_pair = [last_camera_data, camera_data]
            img_pair = np.array(img_pair).transpose(3, 0, 1, 2)
            img_pair = np.expand_dims(img_pair, axis=0)
            img_pair = np.expand_dims(img_pair, axis=0)
            img_pair = torch.from_numpy(img_pair).type(torch.FloatTensor).to('cuda:0')

            self._camera_lock.acquire()
            
            feature_data = self._odometry_model.eval_flownet_model(img_pair)
            self._camera_features.append(feature_data.cpu().numpy())

            self._camera_lock.release()

            tensor_camera_features = torch.from_numpy(np.array(list(self._camera_features), dtype=float)).type(torch.FloatTensor).to('cuda:0')
            tensor_camera_features = torch.squeeze(tensor_camera_features, 1)

            imu_seq = np.expand_dims(imu_seq, 1)
            tensor_imu_seq = torch.from_numpy(imu_seq).type(torch.FloatTensor).to('cuda:0')
            
            self.execute_model(tensor_camera_features, tensor_imu_seq, current_stamp)
            
        
    def execute_model(self, tensor_camera_features, tensor_imu_seq, current_stamp):
        self.get_logger().info('execute')
        self._model_lock.acquire()
        
        if self._beliefs is None:
            beliefs = torch.rand(1,
                                 self._odometry_model.args.belief_size,
                                 device=self._odometry_model.args.device)
        else:
            beliefs = torch.clone(self._beliefs[1, :])

        start_time = time.time()
        self._beliefs, odometry = self._odometry_model.step_model(
            tensor_camera_features,
            tensor_imu_seq,
            self._odometry_state,
            beliefs)
        
        print("--- %s seconds ---" % (time.time() - start_time))
        
        if odometry is not None:
            self._model_lock.release()

            odometry = odometry.cpu().numpy()[-1, 0]
            odometry_msg = Odometry()
            odometry_msg.header.stamp = current_stamp
            odometry_msg.pose.pose.position.x = float(odometry[0])
            odometry_msg.pose.pose.position.y = float(odometry[1])
            odometry_msg.pose.pose.position.z = 0.0
            odometry_msg.pose.pose.orientation.w = 0.0
            odometry_msg.pose.pose.orientation.x = 0.0
            odometry_msg.pose.pose.orientation.y = float(odometry[5])
    #        odometry_msg.pose.pose.orientation.z = odometry[6]

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

            self._publisher.publish(odometry_msg)

    def camera_callback(self, msg):
        self._imu_lock.acquire()
        imu_data = np.array(list(self._imu_data), copy=True)
        self._imu_lock.release()

        if self.frame_nb < self.skip_frame:
            self.frame_nb += 1
            return

        self._imu_seq.append(imu_data)

        camera_data = np.array(list(msg.data), dtype=np.uint8, copy=True)
        last_camera_data = np.array(self._last_camera_data, copy=True) if self._last_camera_data is not None else None
        task = self.executor.create_task(self.process_data, 
                                  camera_data, 
                                  last_camera_data, 
                                  msg.height, 
                                  msg.width, 
                                  np.array(list(self._imu_seq), dtype=float, copy=True), 
                                  msg.header.stamp)
        
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

        self._last_camera_data = np.array(camera_data, copy=True)
        
    def imu_callback(self, msg):
        self._imu_lock.acquire()
        imu_data = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
             msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], copy=True)
        self._imu_data.append(imu_data)
        self._imu_lock.release()

def main(args=None):
    rclpy.init(args=args)

    executor = rclpy.executors.MultiThreadedExecutor(num_threads=8)
    localization_node = P3atDeepvio()
    executor.add_node(localization_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        localization_node.get_logger().info('Shutting down...')
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
