import math
import numpy as np
import info_odometry.sophus as sp
import torch
from timeit import default_timer as timer

from info_odometry.seq_model import SeqVINet
from info_odometry.info_model import SingleHiddenTransitionModel
from info_odometry.info_model import DoubleHiddenTransitionModel
from info_odometry.info_model import SingleHiddenVITransitionModel
from info_odometry.info_model import DoubleHiddenVITransitionModel
from info_odometry.info_model import MultiHiddenVITransitionModel
from info_odometry.info_model import DoubleStochasticTransitionModel
from info_odometry.info_model import DoubleStochasticVITransitionModel
from info_odometry.info_model import ObservationModel
from info_odometry.info_model import PoseModel
from info_odometry.info_model import Encoder
from info_odometry.flownet_model import FlowNet2S
from scipy.spatial.transform import Rotation as R


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, init_lr, d_model, n_warmup_steps=12800):
        self._optimizer = optimizer
        self.param_groups = self._optimizer.param_groups
        self.init_lr = init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def state_dict(self):
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict)

    def step(self):
        "Step with the inner optimizer and update learning_rate"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class SequenceTimer:
    def __init__(self):
        self.curr_time = timer()
        self.avg_time = 0  # average running time for one step
        self.cnt_step = 0
        self.last_time_elapsed = 0

    def tictoc(self):
        curr_time = timer()
        self.last_time_elapsed = curr_time - self.curr_time
        self.curr_time = curr_time
        self.cnt_step += 1
        self.avg_time += (self.last_time_elapsed - self.avg_time) / self.cnt_step

    def get_last_time_elapsed(self):
        return self.last_time_elapsed

    def get_remaining_time(self, curr_step, total_step):
        return (total_step - curr_step) * self.avg_time


class RunningAverager:
    def __init__(self):
        self.avg = 0
        self.cnt = 0

    def append(self, value):
        self.cnt += 1
        self.avg += (value - self.avg) / self.cnt

    def item(self):
        return self.avg

    def cnt(self):
        return self.cnt


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
    euler_result = np.flip(R.from_matrix(transform_result[:3, :3]).as_euler('zyx'),0)
    trans_result = transform_result[:3, 3]
    euler_result[0] = 0.0
    euler_result[1] = 0.0
    trans_result[2] = 0.0

    return np.concatenate((trans_result, euler_result), 0)


# Calculates rotation matrix to euler angles
# The result is for ZYX euler angles
def rotationMatrixToEulerAngles(R):
    # assert(isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def euler_to_quaternion(r, isRad=False):
    if not isRad:
        # By default, isRad is False => r is euler angles in degrees!
        r = r * np.pi / 180
    (yaw, pitch, roll) = (r[2], r[1], r[0])
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw, qx, qy, qz = makeFirstPositive(qw, qx, qy, qz)
    return np.array([qw, qx, qy, qz])


def normalize_imgfeat(img_pair, rgb_max):
    # input -> img_pair: [batch, 3, 2, h, w]
    # output -> img_feature: [batch, 6, 480, 752]
    rgb_mean = img_pair.contiguous().view(img_pair.size()[:2] + (-1,)).mean(dim=-1).view(
        img_pair.size()[:2] + (1, 1, 1,))  # [batch, 3, 1, 1, 1]
    img_pair = (img_pair - rgb_mean) / rgb_max  # [batch, 3, 2, 480, 752], normalized
    x1 = img_pair[:, :, 0, :, :]
    x2 = img_pair[:, :, 1, :, :]
    img_features = torch.cat((x1, x2), dim=1)  # [batch, 6, 480, 752]
    return img_features


def makeFirstPositive(ww, wx, wy, wz):
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


def get_zero_se3():
    """
    output: se3 vector6 for zero movement
    """
    zero_q = sp.Quaternion(1, sp.Vector3(0, 0, 0))
    RT = sp.Se3(sp.So3(zero_q), sp.Vector3(0, 0, 0))
    numpy_vec = np.array(RT.log()).astype(float)
    return np.concatenate(numpy_vec)


def save_model(path, transition_model, pose_model, encoder, optimizer, epoch, metrics, observation_model=None,
               observation_imu_model=None):
    states = {
        'transition_model': transition_model.state_dict(),
        'pose_model': pose_model.state_dict(),
        'encoder': encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }
    if observation_model:
        states['observation_model'] = observation_model.state_dict()
    if observation_imu_model:
        states['observation_imu_model'] = observation_imu_model.state_dict()
    torch.save(states, path)


def construct_models(args):
    # check the validity of soft / hard deepvio
    if args.soft or args.hard:
        if args.transition_model not in ['deepvio', 'double-vinet']:
            raise ValueError('--soft and --hard must be used with deepvio or double-vinet')

    # construct flownet_model
    flownet_model = None
    if args.flownet_model != 'none' and args.img_prefeat == 'none':
        if args.train_img_from_scratch:
            raise ValueError('if --flownet_model -> --train_img_from_scratch should not be used')

        if args.flownet_model == 'FlowNet2S':
            flownet_model = FlowNet2S(args).to(device=args.device)
        else:
            raise ValueError('--flownet_model: {} is not supported'.format(args.flownet_model))
        resume_ckp = '/data/results/ckp/pretrained_flownet/{}_checkpoint.pth.tar'.format(
            args.flownet_model)
        flow_ckp = torch.load(resume_ckp)
        flownet_model.load_state_dict(flow_ckp['state_dict'])
        flownet_model.eval()

    base_args = {
        'args': args,
        'belief_size': args.belief_size,
        'state_size': args.state_size,
        'hidden_size': args.hidden_size,
        'embedding_size': args.embedding_size,
        'activation_function': args.activation_function
    }
    use_imu = False
    use_info = False
    if args.transition_model == 'deepvo':
        transition_model = SeqVINet(use_imu=False, **base_args).to(device=args.device)
    elif args.transition_model == 'deepvio':
        transition_model = SeqVINet(use_imu=True, **base_args).to(device=args.device)
        use_imu = True
    elif args.transition_model == 'single':
        transition_model = SingleHiddenTransitionModel(**base_args).to(device=args.device)
        use_info = True
    elif args.transition_model == 'double':
        transition_model = DoubleHiddenTransitionModel(**base_args).to(device=args.device)
        use_info = True
    elif args.transition_model == 'double-stochastic':
        transition_model = DoubleStochasticTransitionModel(**base_args).to(device=args.device)
        use_info = True
    elif args.transition_model == 'single-vinet':
        transition_model = SingleHiddenVITransitionModel(**base_args).to(device=args.device)
        use_imu = True
        use_info = True
    elif args.transition_model == 'double-vinet':
        transition_model = DoubleHiddenVITransitionModel(**base_args).to(device=args.device)
        use_imu = True
        use_info = True
    elif args.transition_model == 'double-vinet-stochastic':
        transition_model = DoubleStochasticVITransitionModel(**base_args).to(device=args.device)
        use_imu = True
        use_info = True
    elif args.transition_model == 'multi-vinet':
        transition_model = MultiHiddenVITransitionModel(**base_args).to(device=args.device)
        use_imu = True
        use_info = True
    observation_model = None
    observation_imu_model = None
    if args.transition_model not in ['deepvo', 'deepvio']:
        if args.observation_beta != 0: observation_model = ObservationModel(symbolic=True, observation_type='visual',
                                                                            **base_args).to(device=args.device)
        if use_imu and args.observation_imu_beta != 0:
            observation_imu_model = ObservationModel(symbolic=True, observation_type='imu', **base_args).to(
                device=args.device)
    pose_model = PoseModel(**base_args).to(device=args.device)
    encoder = Encoder(symbolic=True, **base_args).to(device=args.device)
    return flownet_model, transition_model, use_imu, use_info, observation_model, observation_imu_model, pose_model, encoder


def eval_rel_error(pred_rel_pose, gt_rel_pose, t_euler_loss):
    """
    pred_rel_pose: predicted se3R6_01 [batch, 6] -> se3 or t_euler
    gt_rel_pose: from se3R6_01 [batch, 6] -> se3 or t_euler
    return:
    -> rpe_all, rpe_trans: [batch] np.sum(array ** 2) -> not sqrt yet
    -> rpe_rot_axis: [batch] anxis-angle (mode of So3.log())
    -> rpe_rot_euler: [batch] np.sum(array ** 2) -> not sqrt yet

    -> v1: from TT'
    -> v2: from ||t-t'||, ||r-r'|| (disabled)
    """
    assert pred_rel_pose.shape[0] == gt_rel_pose.shape[0]
    batch_size = pred_rel_pose.shape[0]
    eval_rel = dict()

    test_rel = dict()
    gt_rel = dict()
    err_rel = dict()

    for _metric in ['x', 'y', 'theta']:
        test_rel[_metric] = []
        gt_rel[_metric] = []
        err_rel[_metric] = []

    for _metric in ['rpe_all', 'rpe_trans', 'rpe_rot_axis', 'rpe_rot_euler']:
        eval_rel[_metric] = []

    for _i in range(batch_size):
        if type(gt_rel_pose) == np.ndarray:
            tmp_pred_rel_pose = pred_rel_pose[_i]
            tmp_gt_rel_pose = gt_rel_pose[_i]
        else:
            tmp_pred_rel_pose = pred_rel_pose[_i].cpu().numpy()
            tmp_gt_rel_pose = gt_rel_pose[_i].cpu().numpy()

        T_01_rel = get_relative_pose(tmp_gt_rel_pose, tmp_pred_rel_pose)

    test_rel['x'].append(pred_rel_pose[-1, 0].cpu().numpy())
    test_rel['y'].append(pred_rel_pose[-1, 1].cpu().numpy())
    test_rel['theta'].append(pred_rel_pose[-1, 5].cpu().numpy())

    gt_rel['x'].append(gt_rel_pose[-1, 0].cpu().numpy())
    gt_rel['y'].append(gt_rel_pose[-1, 1].cpu().numpy())
    gt_rel['theta'].append(gt_rel_pose[-1, 5].cpu().numpy())

    err_rel['x'].append(T_01_rel[0])
    err_rel['y'].append(T_01_rel[1])
    err_rel['theta'].append(T_01_rel[5])

    # each value in eval_rel: a list  with length eval_batch_size
    return eval_rel, test_rel, gt_rel, err_rel


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


def eval_global_error(accu_global_pose, gt_global_pose, test_gt_global_pose):
    """
    input: (list -> batch)
    -> accu_global_pose: list of sp.Se3 Object
    -> gt_global_pose: list of (translation, quaternion) with length 7
    return:
    -> ate_all, ate_trans [batch] -> np.sum(array ** 2)
    -> ate_rot_axis: [batch] axis-angle
    -> ate_rot_euler: [batch] -> np.sum(array ** 2)

    -> v1: from TT'
    -> v2: from ||t-t'||, ||r-r'|| (disabled)
    """
    assert len(accu_global_pose) == len(gt_global_pose)

    eval_global = dict()
    gt_global = dict()
    err_global = dict()
    test_global = dict()

    for _metric in ['x', 'y', 'theta']:
        eval_global[_metric] = []
        gt_global[_metric] = []
        err_global[_metric] = []
        test_global[_metric] = []

    pred = accu_global_pose
    gt = gt_global_pose
    gt_test = test_gt_global_pose
    dt = get_relative_pose_from_transform(gt, pred)

    eval_global['x'].append(pred[0])
    eval_global['y'].append(pred[1])
    eval_global['theta'].append(pred[5])

    gt_global['x'].append(gt[0])
    gt_global['y'].append(gt[1])
    gt_global['theta'].append(gt[5])

    err_global['x'].append(dt[0])
    err_global['y'].append(dt[1])
    err_global['theta'].append(dt[5])

    test_global['x'].append(gt_test[0])
    test_global['y'].append(gt_test[1])
    test_global['theta'].append(gt_test[5])

    # each value in eval_global: a list with length eval_batch_size
    return eval_global, gt_global, err_global, test_global


def get_lr(optimizer):
    """
    currently only support optimizer with one param group
    -> please use multiple optimizers separately for multiple param groups
    """
    lr_list = []
    for param_group in optimizer.param_groups:
        lr_list.append(param_group['lr'])
    assert len(lr_list) == 1
    return lr_list[0]


def factor_lr_schedule(epoch, divide_epochs=[], lr_factors=[]):
    """
    -> divide_epochs need to be sorted
    -> divide_epochs and lr_factors should be one-to-one matched
    """
    assert len(divide_epochs) == len(lr_factors)
    tmp_lr_factors = [1.] + lr_factors
    for _i, divide_epoch in enumerate(divide_epochs):
        idx = _i
        if epoch < divide_epoch:
            break
        idx += 1
    return tmp_lr_factors[idx]


