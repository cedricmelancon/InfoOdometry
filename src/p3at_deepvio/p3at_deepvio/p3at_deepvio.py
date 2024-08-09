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
from info_odometry.utils.tools import get_absolute_pose_step
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
        self._last_position = [29.271869156149368, 129.52834578074683, 0.0, 0.0, 0.0, 0.34944565567934077]
        self._last_camera_data = None
        self._last_stamp = None

        self._img_seq = collections.deque(maxlen=self._odometry_model.clip_length)
        self._imu_seq = collections.deque(maxlen=self._odometry_model.clip_length)
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
        start_time = time.time()
        camera_data = self.image_to_tensor(camera_data, height, width)

        if last_camera_data is not None:
            last_camera_data = self.image_to_tensor(last_camera_data, height, width)

            img_pair = [last_camera_data, camera_data]
            img_pair = np.array(img_pair).transpose(3, 0, 1, 2)
            img_pair = np.expand_dims(img_pair, axis=0)
            img_pair = np.expand_dims(img_pair, axis=0)
            img_pair = torch.from_numpy(img_pair).type(torch.FloatTensor).to('cuda:0')

            feature_data = torch.clone(self._odometry_model.eval_flownet_model(img_pair))

            self._img_seq.append(feature_data.squeeze(0).cpu().numpy())
            imu_seq = np.expand_dims(imu_seq, 0)
            tensor_imu_seq = torch.from_numpy(imu_seq).type(torch.FloatTensor).to('cuda:0')

            self.execute_model(feature_data, tensor_imu_seq, current_stamp)

        #print("--- %s seconds ---" % (time.time() - start_time))

    def execute_model(self, tensor_camera_features, tensor_imu_seq, current_stamp):
        self._model_lock.acquire()
        img_seq = np.array(list(self._img_seq))
        imu_seq = np.array(list(self._imu_seq))
        odometry = self._odometry_model.step_model(
            torch.from_numpy(img_seq).type(torch.FloatTensor).to('cuda:0'),
            torch.from_numpy(imu_seq).type(torch.FloatTensor).to('cuda:0'),
            self._odometry_state)

        if odometry is not None:
            odometry = odometry.cpu().numpy()[0]
            dt = [float(odometry[0]), float(odometry[1]), 0.0, 0.0, 0.0, float(odometry[5])]

            self._last_position = get_absolute_pose_step(dt, self._last_position)
            self._model_lock.release()
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

            self._publisher.publish(odometry_msg)
        else:
            self._model_lock.release()

    def camera_callback(self, msg):
        self._imu_lock.acquire()
        imu_data = np.array(list(self._imu_data), copy=True)
        self._imu_lock.release()

        self._imu_seq.append(np.expand_dims(imu_data, 0))
        if self.frame_nb < self.skip_frame:
            self.frame_nb += 1
            return

        self._camera_lock.acquire()
        camera_data = np.array(list(msg.data), dtype=np.uint8, copy=True)
        self._camera_lock.release()
        last_camera_data = np.array(self._last_camera_data, copy=True) if self._last_camera_data is not None else None
        task = self.executor.create_task(self.process_data,
                                         camera_data,
                                         last_camera_data,
                                         msg.height,
                                         msg.width,
                                         imu_data,
                                         msg.header.stamp)

        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

        self._last_camera_data = np.array(camera_data, copy=True)

    def imu_callback(self, msg):
        self._imu_lock.acquire()
        imu_data = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                             msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
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
