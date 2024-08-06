import rclpy
from rclpy.node import Node
from rclpy.time import Time

import datetime
from nav_msgs.msg import Odometry
import csv
import numpy as np
import math
import os
from threading import Lock


class P3atMetrics(Node):

    def __init__(self):
        super().__init__('P3atMetrics')
        self._csv_lock = Lock()
        self._csv_file = open(os.path.join('/data', 'metrics.csv'), 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file, delimiter=' ')
        self.subscription = self.create_subscription(
            Odometry,
            '/deepvio_odometry',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    @staticmethod
    def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians
    
    def listener_callback(self, msg):
        time = Time.from_msg(msg.header.stamp).seconds_nanoseconds()
        euler = self.euler_from_quaternion(msg.pose.pose.orientation.x,
                                           msg.pose.pose.orientation.y,
                                           msg.pose.pose.orientation.z,
                                           msg.pose.pose.orientation.w)
        data = np.array([time[0] + time[1] * 1e-9, msg.pose.pose.position.x, msg.pose.pose.position.y, euler[2]])
        self._csv_lock.acquire()
        self._csv_writer.writerow(data)
        self._csv_lock.release()


def main(args=None):
    rclpy.init(args=args)

    metrics_subscriber = P3atMetrics()

    rclpy.spin(metrics_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    metrics_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()