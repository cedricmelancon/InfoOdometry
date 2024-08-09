import math
import numpy as np


def _make_first_positive(ww, wx, wy, wz):
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


# Calculates rotation matrix to euler angles
# The result is for ZYX euler angles
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


def euler_to_quaternion(r, is_rad=False):
    if not is_rad:
        # By default, isRad is False => r is euler angles in degrees!
        r = r * np.pi / 180
    (yaw, pitch, roll) = (r[2], r[1], r[0])
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw, qx, qy, qz = _make_first_positive(qw, qx, qy, qz)
    return np.array([qw, qx, qy, qz])
