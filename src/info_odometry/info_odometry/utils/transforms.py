import numpy as np
from scipy.spatial.transform import Rotation as R
import torch


def _init_transforms(euler, translation):
    rotation = R.from_euler('zyx', np.flip(euler, 0)).as_matrix()
    transform = np.zeros((4, 4))
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    transform[3, 3] = 1.0

    return transform


def _get_trans_euler_from_vector(t, force_zero=False):
    translation = t[:3]
    euler = t[-3:]

    if force_zero:
        translation[2] = 0.0
        euler[0] = 0.0
        euler[1] = 0.0

    return translation, euler


def get_relative_pose(t1, t2):
    """ t1 and t2 are vectors of the form [tx ty tz ex ey ez] where t is translation and e is euler angle in rad"""
    trans_t1, euler_t1 = _get_trans_euler_from_vector(t1)
    trans_t2, euler_t2 = _get_trans_euler_from_vector(t2)

    transform_t1 = _init_transforms(euler_t1, trans_t1)
    transform_t2 = _init_transforms(euler_t2, trans_t2)
    transform_result = np.dot(np.linalg.inv(transform_t1), transform_t2)
    euler_result = np.flip(R.from_matrix(transform_result[:3, :3]).as_euler('zyx'), 0)
    trans_result = transform_result[:3, 3]
    euler_result[0] = 0.0
    euler_result[1] = 0.0
    trans_result[2] = 0.0

    return np.concatenate((trans_result, euler_result), 0)


def get_absolute_pose_step(dt, state):
    trans_state, euler_state = _get_trans_euler_from_vector(state, True)
    trans_dt, euler_dt = _get_trans_euler_from_vector(dt, True)

    transform_state = _init_transforms(euler_state, trans_state)
    transform_dt = _init_transforms(euler_dt, trans_dt)

    transform_result = np.dot(transform_state, transform_dt)

    euler_result = R.from_matrix(transform_result[:3, :3]).as_quat()
    trans_result = transform_result[:3, 3]

    return np.concatenate((trans_result, euler_result), 0)


def get_absolute_pose(dt, state):
    clip_size = dt.size()[0]
    result = [torch.empty(0)] * clip_size

    last_state = state.squeeze(0).squeeze(0).cpu().numpy()
    for i in range(clip_size):
        if i > 0:
            last_state = result[i - 1]

        value = get_absolute_pose_step(dt[i].squeeze(0).cpu().numpy(), last_state)
        result[i] = value

    return result
