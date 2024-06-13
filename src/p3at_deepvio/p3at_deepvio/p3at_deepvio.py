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


class P3atDeepvio(Node):
    def __init__(self):
        super().__init__('P3atDeepvio')

        self._imu_lock = Lock()

        param = Param()
        args = param.get_args()
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        self._odometry_model = OdometryModel(args)
        self._publisher = self.create_publisher(Odometry, 'deepvio_odometry', 10)
        self._imu_subscriber = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self._camera_subscriber = self.create_subscription(Image, '/camera/rgb/image_raw', self.camera_callback, 10)
        self._imu_data = []
        self._imu_seq = collections.deque(maxlen=5)
        self._camera_features = collections.deque(maxlen=5)
        self._last_position = None
        self._last_camera_data = None
        self._last_stamp = None
        self.get_logger().info('C\'est parti!')
        self._odometry_state = torch.zeros(1,
                                           1024) #,  # args.state_size
                                           #device=args.device)
        self._beliefs = torch.rand(1,
                                   1024) # args.belief_size
                                   # device=args.device)

    @staticmethod
    def image_to_tensor(image, width, height):
        image = image.reshape(1, height, width).transpose(1, 2, 0)
        image = cv2.cvtColor(image, 88)
        image = np.array(Img.fromarray(image))

        return torch.from_numpy(image).type(torch.FloatTensor).to('cuda:0')

    def camera_callback(self, msg):
        self._imu_lock.acquire()
        self._imu_seq.append(torch.from_numpy(np.array(self._imu_data, copy=True)).to('cuda:0'))
        self._imu_data = []
        self._imu_lock.release()

        camera_data = np.array(list(msg.data), dtype=np.uint8)
        camera_data = self.image_to_tensor(camera_data, msg.height, msg.width)
        print(camera_data.shape)

        if self._last_camera_data is not None:
            img_pair = torch.stack((self._last_camera_data, camera_data), dim=1).type(torch.FloatTensor).to('cuda:0')
            print(img_pair.shape)
            feature_data = self._odometry_model.eval_flownet_model(img_pair)
            self._camera_features.append(feature_data)

        self._last_camera_data = torch.clone(camera_data).to('cuda:0')

        current_stamp = msg.header.stamp
        if len(self._camera_features) >= 5:
            self._beliefs, _, _, _, _, _, _, odometry = self._odometry_model.run_eval(
                torch.from_numpy(np.array(list(self._camera_features))).type(torch.FloatTensor),
                torch.from_numpy(np.array(list(self._imu_seq))).type(torch.FloatTensor),
                self._odometry_state,
                self._beliefs)

            odometry_msg = Odometry()
            odometry_msg.header.stamp = current_stamp
            odometry_msg.pose.pose.position.x = odometry[0]
            odometry_msg.pose.pose.position.y = odometry[1]
            odometry_msg.pose.pose.position.z = odometry[2]
            odometry_msg.pose.orientation.w = odometry[3]
            odometry_msg.pose.orientation.x = odometry[4]
            odometry_msg.pose.orientation.y = odometry[5]
            odometry_msg.pose.orientation.z = odometry[6]

            if self._last_position is None:
                odometry_msg.twist.linear.x = 0.0
                odometry_msg.twist.linear.y = 0.0
                odometry_msg.twist.linear.z = 0.0

                odometry_msg.twist.angular.x = 0.0
                odometry_msg.twist.angular.y = 0.0
                odometry_msg.twist.angular.z = 0.0
            else:
                odometry_msg.twist.linear.x = 0.0
                odometry_msg.twist.linear.y = 0.0
                odometry_msg.twist.linear.z = 0.0

                odometry_msg.twist.angular.x = 0.0
                odometry_msg.twist.angular.y = 0.0
                odometry_msg.twist.angular.z = 0.0

    def imu_callback(self, msg):
        self._imu_lock.acquire()
        imu_data = torch.from_numpy(np.array(
            [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
             msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]))
        self._imu_data.append(imu_data)
        self._imu_lock.release()


def main(args=None):
    rclpy.init(args=args)

    localization_node = P3atDeepvio()

    rclpy.spin(localization_node)

    localization_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
