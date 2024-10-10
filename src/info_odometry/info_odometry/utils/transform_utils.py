import torch
from scipy.spatial.transform import Rotation as R
import numpy as np
import math


class TransformUtils:
    def __init__(self):
        pass

    @staticmethod
    def get_relative_pose(t1, t2):
        trans_t1 = t1[:3]
        euler_t1 = t1[-3:]

        trans_t2 = t2[:3]
        euler_t2 = t2[-3:]

        rotation_t1 = R.from_euler('zyx', np.flip(euler_t1, 0)).as_matrix()

        transform_t1 = np.zeros((4, 4))
        transform_t1[:3, :3] = rotation_t1
        transform_t1[:3, 3] = trans_t1
        transform_t1[3, 3] = 1.0

        rotation_t2 = R.from_euler('zyx', np.flip(euler_t2, 0)).as_matrix()

        transform_t2 = np.zeros((4, 4))
        transform_t2[:3, :3] = rotation_t2
        transform_t2[:3, 3] = trans_t2
        transform_t2[3, 3] = 1.0

        transform_result = np.dot(np.linalg.inv(transform_t1), transform_t2)
        euler_result = np.flip(R.from_matrix(transform_result[:3, :3]).as_euler('zyx'), 0)
        trans_result = transform_result[:3, 3]
        euler_result[0] = 0.0
        euler_result[1] = 0.0
        trans_result[2] = 0.0

        return np.concatenate((trans_result, euler_result), 0)

    # Calculates rotation matrix to euler angles
    # The result is for ZYX euler angles
    @staticmethod
    def rotation_matrix_to_euler_angles(rotation):
        # assert(isRotationMatrix(R))
        sy = math.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(rotation[2, 1], rotation[2, 2])
            y = math.atan2(-rotation[2, 0], sy)
            z = math.atan2(rotation[1, 0], rotation[0, 0])
        else:
            x = math.atan2(-rotation[1, 2], rotation[1, 1])
            y = math.atan2(-rotation[2, 0], sy)
            z = 0
        return np.array([x, y, z])

    @staticmethod
    def euler_to_quaternion(rotation, is_rad=False):
        if not is_rad:
            # By default, isRad is False => r is euler angles in degrees!
            rotation = rotation * np.pi / 180
        (yaw, pitch, roll) = (rotation[2], rotation[1], rotation[0])
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
            yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
            yaw / 2)
        qw, qx, qy, qz = make_first_positive(qw, qx, qy, qz)
        return np.array([qw, qx, qy, qz])

    @staticmethod
    def make_first_positive(ww, wx, wy, wz):
        """
        make the first non-zero element in q positive
        q_array = [ww, wx, wy, wz]
        """
        q_array = [ww, wx, wy, wz]
        for q_ele in q_array:
            if q_ele == 0:
                continue
            if q_ele < 0:
                q_array = [-x for x in q_array]
            break
        return q_array[0], q_array[1], q_array[2], q_array[3]

    @staticmethod
    def get_absolute_pose_step(dt, state):
        trans_state = state[:3]
        trans_state[2] = 0.0
        euler_state = state[-3:]
        euler_state[0] = 0.0
        euler_state[1] = 0.0

        trans_dt = dt[:3]
        trans_dt[2] = 0.0
        euler_dt = dt[-3:]
        euler_dt[0] = 0.0
        euler_dt[1] = 0.0

        rotation_state = R.from_euler('zyx', np.flip(euler_state, 0)).as_matrix()

        transform_state = np.zeros((4, 4))
        transform_state[:3, :3] = rotation_state
        transform_state[:3, 3] = trans_state
        transform_state[3, 3] = 1.0

        rotation_dt = R.from_euler('zyx', np.flip(euler_dt, 0)).as_matrix()

        transform_dt = np.zeros((4, 4))
        transform_dt[:3, :3] = rotation_dt
        transform_dt[:3, 3] = trans_dt
        transform_dt[3, 3] = 1.0

        transform_result = np.dot(transform_state, transform_dt)

        euler_result = np.flip(R.from_matrix(transform_result[:3, :3]).as_quat(), 0)
        trans_result = transform_result[:3, 3]
        euler_result[0] = 0.0
        euler_result[1] = 0.0
        trans_result[2] = 0.0

        return np.concatenate((trans_result, euler_result), 0)

    @staticmethod
    def get_absolute_pose(dt, state):
        clip_size = dt.size()[0]
        result = [torch.empty(0)] * clip_size

        last_state = state.squeeze(0).squeeze(0).cpu().numpy()
        for i in range(clip_size):
            if i > 0:
                last_state = result[i - 1]

            value = TransformUtils.get_absolute_pose_step(dt[i].squeeze(0).cpu().numpy(), last_state)
            result[i] = value

        return result

    @staticmethod
    def get_relative_pose_from_transform(t1, t2):
        trans_t1 = t1[:3]
        trans_t1[2] = 0.0
        euler_t1 = t1[-3:]
        euler_t1[0] = 0.0
        euler_t1[1] = 0.0

        trans_t2 = t2[:3]
        trans_t2[2] = 0.0
        euler_t2 = t2[-3:]
        euler_t2[0] = 0.0
        euler_t2[1] = 0.0

        rotation_t1 = R.from_euler('zyx', np.flip(euler_t1, 0)).as_matrix()
        transform_t1 = np.zeros((4, 4))
        transform_t1[:3, :3] = rotation_t1
        transform_t1[:3, 3] = trans_t1
        transform_t1[3, 3] = 1.0

        rotation_t2 = R.from_euler('zyx', np.flip(euler_t2, 0)).as_matrix()

        transform_t2 = np.zeros((4, 4))
        transform_t2[:3, :3] = rotation_t2
        transform_t2[:3, 3] = trans_t2
        transform_t2[3, 3] = 1.0

        transform_result = np.dot(np.linalg.inv(transform_t1), transform_t2)
        euler_result = np.flip(R.from_matrix(transform_result[:3, :3]).as_euler('zyx'), 0)
        trans_result = transform_result[:3, 3]
        euler_result[0] = 0.0
        euler_result[1] = 0.0
        trans_result[2] = 0.0

        return np.concatenate((trans_result, euler_result), 0)