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
from info_odometry.utils.transform_utils import TransformUtils

#from BlazeAIoT.Core.NodeManager import ServiceManager
import time
import csv

from p3at_deepvio.monitor import Monitor

class P3atDeepvio(Node):
    def __init__(self):
        super().__init__('P3atDeepvio')
        #self._service_manager = ServiceManager()

        self._monitor = Monitor(cuda_enabled=True)
        self.beliefs = None
        self._imu_lock = Lock()
        self._camera_lock = Lock()
        self._model_lock = Lock()
        self._flownet_lock = Lock()
        self._monitoring_lock = Lock()
        self._sys_mon_lock = Lock()

        self.skip_frame = 5
        self.frame_nb = 0
        self.timing_monitor = {}

        param = Param()
        self.args = param.get_args()
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)

        publisher_group = ReentrantCallbackGroup()
        subscriber1_group = ReentrantCallbackGroup()
        subscriber2_group = ReentrantCallbackGroup()
        timer_group = MutuallyExclusiveCallbackGroup()
        monitoring_group = MutuallyExclusiveCallbackGroup()

        self._odometry_model = OdometryModel(self.args)

        self._publisher = self.create_publisher(Odometry, '/odometry', 10, callback_group=publisher_group)
        self._imu_subscriber = self.create_subscription(Imu, '/torso_lift_imu/data', self.imu_callback, 10,
                                                        callback_group=subscriber1_group)
        self._camera_subscriber = self.create_subscription(Image, '/camera/rgb/image_raw', self.camera_callback, 10,
                                                           callback_group=subscriber2_group)
        self._imu_data = collections.deque(maxlen=3)
        self._last_position = [29.271869156149368, 129.52834578074683, 0.0, 0.0, 0.0, 0.34944565567934077]
        self._last_camera_data = None
        self._last_stamp = None

        #self._img_seq = torch.zeros([2, 81920]).to('cuda:0')
        self._monitoring_data = collections.deque()
        self._odometry_state = torch.zeros(1,
                                           self._odometry_model.args.state_size,
                                           device=self._odometry_model.args.device)
        self._beliefs = None
        self._monitoring_task = None

        self.prev_beliefs = torch.rand(1, self.args.belief_size, device=self.args.device)
        self.prev_state = torch.zeros(1, self.args.state_size, device=self.args.device)

        csvfile = open(f'monitoring.csv', 'w', newline='')
        self.csvwriter = csv.writer(csvfile, delimiter=' ')

        mon_csvfile = open(f'system.csv', 'w', newline='')
        self.system_csv_writer = csv.writer(mon_csvfile, delimiter=' ')

        self.timer = self.create_timer(0.3, self.write_timing, callback_group=timer_group)
        self.timer = self.create_timer(0.1, self.write_system_info, callback_group=monitoring_group)
        self.get_logger().info('Running')

    def write_system_info(self):
        self._sys_mon_lock.acquire()
        cpu = self._monitor.get_cpu_info()["cpu/load/avg_sys_load_one_min_percent"]
        ram = self._monitor.get_memory_info()
        ram_avail = ram["memory/available_memory_sys_MB"]
        ram_used = ram["memory/used_memory_sys_MB"]
        ram_perc = ram["memory/used_memory_sys_percent"]
        smi = self._monitor.get_nvidia_smi_info()
        gpu_avail = smi["05_gpu_smi/gpu_0_fb_total_MiB"]
        gpu_used = smi["05_gpu_smi/gpu_0_fb_used_MiB"]
        gpu_perc = smi["05_gpu_smi/gpu_0_fb_free_MiB"]
        gpu_temp = smi["05_gpu_smi/gpu_0_temp_in_C"]
        gpu_power = smi["05_gpu_smi/gpu_0_power_in_W"]

        data = np.array([cpu, ram_avail, ram_used, ram_perc, gpu_avail, gpu_used, gpu_perc, gpu_temp, gpu_power])
        self.system_csv_writer.writerow(data)
        self._sys_mon_lock.release()

    def write_timing(self):
        if len(self._monitoring_data) > 0:
            self._monitoring_lock.acquire()
            data = self._monitoring_data.pop()
            self._monitoring_lock.release()
            self.csvwriter.writerow(data)

    @staticmethod
    def image_to_tensor(image, width, height):
        image = image.reshape(height, width)
        image = cv2.cvtColor(image, cv2.COLOR_BAYER_GR2BGR)

        return image.transpose(1, 0, 2)

    @staticmethod
    def push_to_tensor_alternative(tensor, x):
        return torch.cat((tensor[1:7], x))

    def process_data(self, camera_data, last_camera_data, imu_data, height, width, current_stamp):
        camera_data = self.image_to_tensor(camera_data, height, width)

        if last_camera_data is not None:
            self._flownet_lock.acquire()
            start_time = time.perf_counter()
            last_camera_data = self.image_to_tensor(last_camera_data, height, width)
            
            img_pair = [last_camera_data, camera_data]
            img_pair = np.array(img_pair).transpose(3, 0, 1, 2)
            img_pair = np.expand_dims(img_pair, axis=0)
            img_pair = np.expand_dims(img_pair, axis=0)
            img_pair = torch.from_numpy(img_pair).type(torch.FloatTensor).to('cuda:0')

            feature_data = self._odometry_model.forward_flownet(img_pair)

            flownet_time = (time.perf_counter() - start_time)
            self._flownet_lock.release()

            self._model_lock.acquire()
            clone_feature = torch.clone(feature_data)
            
            if self.beliefs is None:
                self.beliefs = torch.rand(1, self.args.belief_size, device=self.args.device)

            imu_seq = torch.from_numpy(imu_data).type(torch.FloatTensor).to('cuda:0')
            imu_seq = imu_seq.unsqueeze(0)
            prev_beliefs = self.beliefs

            with torch.no_grad():
                self.beliefs, odometry, timing = self._odometry_model.step(clone_feature, imu_seq, prev_beliefs)

            self._model_lock.release()

            self._monitoring_lock.acquire()
            self._monitoring_data.append(
                np.array([self.frame_nb, flownet_time] + timing))
            self._monitoring_lock.release()

            if odometry is not None:
                odometry = odometry.cpu().numpy()[0]

                dt = [float(odometry[0]), float(odometry[1]), 0.0, 0.0, 0.0, float(odometry[5])]

                self._last_position = TransformUtils.get_absolute_pose_step(dt, self._last_position)
                odometry_msg = Odometry()
                odometry_msg.header.stamp = current_stamp
                odometry_msg.pose.pose.position.x = float(self._last_position[0])
                odometry_msg.pose.pose.position.y = float(self._last_position[1])
                odometry_msg.pose.pose.position.z = 0.0

                euler = np.array(odometry[-3:])
                quat = TransformUtils.euler_to_quaternion(euler)
                odometry_msg.pose.pose.orientation.x = float(quat[0])
                odometry_msg.pose.pose.orientation.y = float(quat[1])
                odometry_msg.pose.pose.orientation.z = float(quat[2])
                odometry_msg.pose.pose.orientation.w = float(quat[3])

                # if self._last_position is None:
                #    odometry_msg.twist.linear.x = 0.0
                #    odometry_msg.twist.linear.y = 0.0
                #    odometry_msg.twist.linear.z = 0.0

                #    odometry_msg.twist.angular.x = 0.0
                #    odometry_msg.twist.angular.y = 0.0
                #    odometry_msg.twist.angular.z = 0.0
                # else:
                #    odometry_msg.twist.linear.x = 0.0
                #    odometry_msg.twist.linear.y = 0.0
                #    odometry_msg.twist.linear.z = 0.0

                #    odometry_msg.twist.angular.x = 0.0
                #    odometry_msg.twist.angular.y = 0.0
                #    odometry_msg.twist.angular.z = 0.0

                # self.rate_counter.inc()
                self._publisher.publish(odometry_msg)

    def camera_callback(self, msg):
        self._imu_lock.acquire()
        imu_data = np.array(list(self._imu_data), copy=True)
        self._imu_lock.release()

        if self.frame_nb < self.skip_frame:
            self.frame_nb += 1
            return

        self.frame_nb += 1
        self._camera_lock.acquire()
        camera_data = np.array(list(msg.data), dtype=np.uint8, copy=True)
        self._camera_lock.release()
        last_camera_data = np.array(self._last_camera_data, copy=True) if self._last_camera_data is not None else None
        self.process_data(camera_data,
                          last_camera_data,
                          imu_data,
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
        pass
        #localization_node.write_timing()
        # localization_node.destroy_node()
        # rclpy.shutdown()


if __name__ == '__main__':
    main()
