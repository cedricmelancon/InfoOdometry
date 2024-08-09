import numpy as np
from transforms import get_relative_pose


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
    dt = get_relative_pose(gt, pred)

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