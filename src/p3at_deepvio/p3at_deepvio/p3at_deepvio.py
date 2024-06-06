import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu

from threading import Lock
import numpy as np
from info_odometry.odometry_model import OdometryModel


class P3atDeepvio(Node):
    def __init__(self):
        super().__init__('P3atDeepvio')

        self._imu_lock = Lock()

        self._publisher = self.create_publisher * (Odometry, 'deepvio_odometry', 10)
        self._camera_subscriber = self.create_subscription(Image, 'camera_img_tbd', self.camera_callback, 10)
        self._imu_subscriber = self.create_subscription(Imu, 'imu', self.imu_callback, 10)
        self._imu_data = []
        self._camera_features = []
        self._last_position = None
        self._last_stamp = None

    def camera_callback(self, msg):
        self._imu_lock.acquire()
        imu_data = np.array(self._imu_data, copy=True)
        self._imu_data = []
        self._imu_lock.release()

        camera_data = np.array(msg.data.data)
        camera_data = np.resize(camera_data, (1, msg.data.height, msg.data.width))
        feature_data = camera_data  # TODO appeler FlowNet
        self._camera_features.append(feature_data)

        current_stamp = msg.data.header.stamp
        if len(self._camera_features) >= 5:
            self._camera_features = np.array(self._camera_features[-5:]).tolist()

            # TODO appeler deepvio
            odometry = np.array(7)

            odometry_msg = Odometry()
            odometry_msg.data.header.stamp = current_stamp
            odometry_msg.data.pose.pose.position.x = odometry[0]
            odometry_msg.data.pose.pose.position.y = odometry[1]
            odometry_msg.data.pose.pose.position.z = odometry[2]
            odometry_msg.data.pose.orientation.w = odometry[3]
            odometry_msg.data.pose.orientation.x = odometry[4]
            odometry_msg.data.pose.orientation.y = odometry[5]
            odometry_msg.data.pose.orientation.z = odometry[6]

            if self._last_position is None:
                odometry_msg.data.twist.linear.x = 0.0
                odometry_msg.data.twist.linear.y = 0.0
                odometry_msg.data.twist.linear.z = 0.0

                odometry_msg.data.twist.angular.x = 0.0
                odometry_msg.data.twist.angular.y = 0.0
                odometry_msg.data.twist.angular.z = 0.0
            else:
                odometry_msg.data.twist.linear.x = 0.0
                odometry_msg.data.twist.linear.y = 0.0
                odometry_msg.data.twist.linear.z = 0.0

                odometry_msg.data.twist.angular.x = 0.0
                odometry_msg.data.twist.angular.y = 0.0
                odometry_msg.data.twist.angular.z = 0.0

    def imu_callback(self, msg):
        self._imu_lock.acquire()
        imu_data = np.array(
            [msg.data.linear_acceleration.x, msg.data.linear_acceleration.y, msg.data.linear_acceleration.z,
             msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
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
