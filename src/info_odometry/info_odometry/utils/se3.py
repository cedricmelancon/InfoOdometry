import info_odometry.sophus as sp
import numpy as np


def get_zero_se3():
    """
    output: se3 vector6 for zero movement
    """
    zero_q = sp.Quaternion(1, sp.Vector3(0, 0, 0))
    RT = sp.Se3(sp.So3(zero_q), sp.Vector3(0, 0, 0))
    numpy_vec = np.array(RT.log()).astype(float)
    return np.concatenate(numpy_vec)
