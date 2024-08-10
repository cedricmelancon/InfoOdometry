import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
import csv
from timeit import default_timer as timer
from tqdm import tqdm

import numpy as np

from info_odometry.utils import SequenceTimer
from info_odometry.utils import RunningAverager
from info_odometry.utils import ScheduledOptim

from info_odometry.utils.tools import save_model, get_lr
from info_odometry.utils.odom_model import eval_rel_error, eval_global_error
from info_odometry.utils.transforms import get_absolute_pose

from info_odometry.param import Param
from info_odometry.odometry_model import OdometryModel

from info_odometry.dataset.mit_stata_center_dataset import load_mit_clips


def save_data(writer, loss, labels_global, labels_delta, pred_abs, pred_rel, epoch, n_iter):
    row = np.concatenate((np.array([epoch, n_iter]),
                          loss.cpu().detach().numpy(),
                          labels_global.cpu().detach().numpy()[-1, :],
                          labels_delta.cpu().detach().numpy()[-1, :],
                          pred_rel.cpu().detach().numpy()[-1, :],
                          np.array(pred_abs)[-1,:]), axis=None)
    writer.writerow(row)


def train(args):
    """
    args: see param.py for details
    """
    torch.cuda.manual_seed(args.seed)
    epoch = args.epoch
    writer = SummaryWriter(log_dir='{}{}/'.format(args.tb_dir, args.exp_name))
    eval_csvfile = open(f'{args.tb_dir}/{args.exp_name}/test_data.csv', 'w', newline='')
    eval_csvwriter = csv.writer(eval_csvfile, delimiter=' ')

    model = OdometryModel(args)

    if args.load_model != "none":
        model.load_model(args.load_model)
    if args.lr_warmup:
        optimizer = ScheduledOptim(
            optimizer=optim.Adam(model.param_list, betas=(0.9, 0.98), eps=1e-09),
            init_lr=args.lr,
            d_model=args.belief_size,
            n_warmup_steps=args.n_warmup_steps
        )
    else:
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.param_list, lr=args.lr, momentum=args.momentum)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.param_list, lr=args.lr, eps=args.adam_epsilon)
        elif args.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(model.param_list, lr=args.lr, eps=args.adam_epsilon)
        else:
            raise ValueError('optimizer {} is currently not supported'.format(args.optimizer))

    if not args.lr_warmup:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=10, threshold_mode='abs')

    # initialize datasets and data loaders
    train_clips = load_mit_clips(
        seqs=args.train_sequences,
        seqs_gt=args.train_sequences_gt,
        batch_size=args.batch_size,
        shuffle=True,
        overlap=args.clip_overlap,
        args=args,
        sample_size_ratio=args.sample_size_ratio
    )

    eval_clips = load_mit_clips(
        seqs=args.eval_sequences,
        seqs_gt=args.eval_sequences_gt,
        batch_size=args.eval_batch_size,
        shuffle=False,
        overlap=True,
        args=args,
        sample_size_ratio=1.
    )

    best_epoch = {'sqrt_then_avg': 0}  # 'avg_then_sqrt': 0
    best_eval = {'total_loss': 100000.0}  # 'avg_then_sqrt': 1.0
    best_metrics = {'total_loss': None}  # 'avg_then_sqrt': None

    odometry_model = OdometryModel(args)

    # starting training (the same epochs for each sequence)
    curr_iter = 0
    curr_eval_iter = 0

    for epoch_idx in range(epoch):
        print('-----------------------------------------')
        print('starting epoch {}...'.format(epoch_idx))
        print('learning rate: {:.6f}'.format(get_lr(optimizer)))
        writer.add_scalar('general/learning_rate', get_lr(optimizer), epoch_idx)

        batch_timer = SequenceTimer()

        for batch_idx, batch_data in tqdm(enumerate(train_clips)):
            # x_img_list:                length-5 list with component [batch, 3, 2, H, W]
            # x_imu_list:                length-5 list with component [batch, 11, 6]
            # x_last_rel_pose_list:      length-5 list with component [batch, 6]    # se3 or t_euler (--t_euler_loss)
            # y_rel_pose_list:           length-5 list with component [batch, 6]    # se3 or t_euler (--t_euler_loss)
            # y_last_global_pose_list:   length-5 list with component [batch, 7]    # t_quaternion
            # y_global_pose_list:        length-5 list with component [batch, 7]    # t_quaternion
            (x_img_list, x_imu_list,
             x_last_rel_pose_list, y_rel_pose_list,
             y_last_global_pose_list, y_global_pose_list, _, _) = batch_data
            
            x_img_pairs = torch.stack(x_img_list, dim=0).type(torch.FloatTensor).to(device=args.device) # [time, batch, 3, 2, H, W]
            y_rel_poses = torch.stack(y_rel_pose_list, dim=0).type(torch.FloatTensor).to(device=args.device) # [time, batch, 6]

            x_imu_seqs = torch.stack(x_imu_list, dim=0).type(torch.FloatTensor).to(device=args.device) # [time, batch, 11, 6]
            running_batch_size = x_img_pairs.size()[1] # might be different for the last batch

            # transitions start at time t = 0
            # create initial belief and state for time t = 0
            init_belief = torch.rand(running_batch_size, args.belief_size, device=args.device)

            beliefs, pred_rel_poses = odometry_model.run_train(x_img_pairs,
                                                               x_imu_seqs,
                                                               init_belief)

            (pose_rot_loss, pose_trans_loss_x,
             pose_trans_loss_y, total_loss) = compute_loss(args, pred_rel_poses, y_rel_poses)

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm(odometry_model.param_list, args.grad_clip_norm, norm_type=2)
            optimizer.step()  # if using ScheduledOptim -> will also update learning rate

            writer.add_scalar('train/total_loss', total_loss.item(), curr_iter)
            writer.add_scalar('train/pose_trans_loss_x', pose_trans_loss_x.item(), curr_iter)
            writer.add_scalar('train/pose_trans_loss_y', pose_trans_loss_y.item(), curr_iter)
            writer.add_scalar('train/pose_rot_loss', pose_rot_loss.item(), curr_iter)
            writer.add_scalar('train/learning_rate', get_lr(optimizer), curr_iter)
            curr_iter += 1

            batch_timer.tictoc()

        # evaluate the model after training each sequence
        # if gt_last_pose is False, then zero_first must be True
        if epoch_idx % args.eval_interval == 0:
            # move eval directly here
            print('----------------------------------------')
            batch_timer = SequenceTimer()

            loss_avg = dict()
            loss_list = ['total_loss', 'pose_trans_loss_x', 'pose_trans_loss_y', 'pose_rot_loss']
            last_pose = None
            last_gt_pose = None

            for _met in loss_list:
                loss_avg[_met] = RunningAverager()

            beliefs = None
            for batch_idx, batch_data in tqdm(enumerate(eval_clips)):
                if args.debug and batch_idx >= 10:
                    break

                x_img_list, \
                    x_imu_list, \
                    x_last_rel_pose_list, \
                    y_rel_pose_list, \
                    y_last_global_pose_list, \
                    y_global_pose_list, _, _ = batch_data

                # [time, batch, 3, 2, H, W]
                x_img_pairs = torch.stack(x_img_list, dim=0).type(torch.FloatTensor).to(device=args.device)
                y_rel_poses = torch.stack(y_rel_pose_list, dim=0).type(torch.FloatTensor).to(device=args.device)
                y_glob_poses = (torch.stack(y_last_global_pose_list, dim=0)
                                .type(torch.FloatTensor)
                                .to(device=args.device))

                if last_pose is None:
                    last_pose = y_glob_poses[0].unsqueeze(0)
                    last_gt_pose = y_glob_poses[0].unsqueeze(0)

                x_imu_seqs = torch.stack(x_imu_list, dim=0).type(torch.FloatTensor).to(device=args.device)

                running_eval_batch_size = x_img_pairs.size()[1] # might be different at the last batch
                init_state = torch.zeros(running_eval_batch_size, args.state_size, device=args.device)

                if beliefs is None:
                    beliefs = torch.rand(running_eval_batch_size, args.belief_size, device=args.device)
                else:
                    beliefs = beliefs[1, :]

                observations = odometry_model.eval_flownet_model(x_img_pairs)

                beliefs, pred_rel_poses = odometry_model.run_eval(observations, x_imu_seqs, beliefs)

                (pose_rot_loss, pose_trans_loss_x,
                 pose_trans_loss_y, total_loss) = compute_loss(args, pred_rel_poses, y_rel_poses)

                loss_avg['total_loss'].append(total_loss)
                loss_avg['pose_trans_loss_x'].append(pose_trans_loss_x)
                loss_avg['pose_trans_loss_y'].append(pose_trans_loss_y)
                loss_avg['pose_rot_loss'].append(pose_rot_loss)

                for _fidx in range(args.clip_length):
                    # (1) evaluate relative pose error (2) no discard_num is used
                    (eval_rel, test_rel, gt_rel, err_rel) = eval_rel_error(pred_rel_poses[_fidx],
                                                                           y_rel_poses[_fidx],
                                                                           t_euler_loss=args.t_euler_loss)

                new_pose = get_absolute_pose(pred_rel_poses, last_pose)
                new_gt_pose = get_absolute_pose(y_rel_poses, last_gt_pose)

                (eval_glob, gt_glob,
                 err_glob, test_glob) = eval_global_error(new_pose[-1],
                                                          y_global_pose_list[-1].squeeze(0).cpu().numpy(),
                                                          new_gt_pose[-1])

                for _met in ['x', 'y', 'theta']:
                    writer.add_scalars(f'test/abs_{_met}',
                                       {
                                           'eval': eval_glob[_met],
                                           'gt': gt_glob[_met],
                                           'test': test_glob[_met]},
                                       batch_idx)
                    writer.add_scalar(f'test/abs_err_{_met}', err_glob[_met], batch_idx)
                    writer.add_scalars(f'test/rel_{_met}',
                                       {
                                           'eval': test_rel[_met],
                                           'gt': gt_rel[_met]},
                                       batch_idx)
                    writer.add_scalar(f'test/rel_err_{_met}', err_rel[_met], batch_idx)

                writer.add_scalar('eval/pose_trans_loss_x', pose_trans_loss_x.item(), curr_eval_iter)
                writer.add_scalar('eval/pose_trans_loss_y', pose_trans_loss_y.item(), curr_eval_iter)
                writer.add_scalar('eval/pose_rot_loss', pose_rot_loss.item(), curr_eval_iter)
                curr_eval_iter += 1
                save_data(eval_csvwriter,
                          total_loss,
                          y_glob_poses,
                          y_rel_poses,
                          new_pose,
                          pred_rel_poses,
                          epoch_idx,
                          batch_idx)
                last_pose = torch.from_numpy(new_pose[0])
                last_gt_pose = torch.from_numpy(new_gt_pose[0])

                batch_timer.tictoc()

            out_eval = dict()
            for _loss in loss_list:
                out_eval[_loss] = loss_avg[_loss].item()
                writer.add_scalar('eval/{}_'.format(_loss), out_eval[_loss], epoch_idx)

            # update learning rate for next epoch
            if not args.lr_warmup:
                scheduler.step(loss_avg['total_loss'].item())

            check_str = '{}{}/ckp_latest.pt'.format(args.ckp_dir, args.exp_name)
            save_args = {
                'transition_model': odometry_model.transition_model,
                'observation_model': odometry_model.observation_model, # None if not used
                'observation_imu_model': odometry_model.observation_imu_model, # None if not used
                'pose_model': odometry_model.pose_model,
                'encoder': odometry_model.encoder,
                'optimizer': optimizer,
                'epoch': epoch_idx,
                'metrics': out_eval
            }

            save_model(path=check_str, **save_args)
            if epoch_idx > 198 and epoch_idx % 100 == 0:
                check_str = '{}{}/ckp_epoch_{}.pt'.format(args.ckp_dir, args.exp_name, epoch_idx + 1)
                save_model(path=check_str, **save_args)

            if out_eval['total_loss'] < best_eval['total_loss'] or best_metrics['total_loss'] is None:
                best_eval['total_loss'] = out_eval['total_loss']
                best_epoch['total_loss'] = epoch_idx
                best_metrics = out_eval
                check_str = '{}{}/ckp_best-eval-loss.pt'.format(args.ckp_dir, args.exp_name)
                save_model(path=check_str, **save_args)
            
            print('====================================')
            print('current epoch for sqrt_then_avg')
            print('====================================')
            loss_str = 'pose_trans_loss: {:.5f} | pose_rot_loss: {:.5f}'.format(out_eval['pose_trans_loss_x'] + out_eval['pose_trans_loss_y'], out_eval['pose_rot_loss'])
            print('eval epoch: {} | total_loss: {:.5f} | {}'.format(epoch_idx, out_eval['total_loss'], loss_str))

            print('====================================')
            print('best epoch for sqrt_then_avg')
            print('====================================')
            loss_str = 'pose_trans_loss: {:.5f} | pose_rot_loss: {:.5f}'.format(
                best_metrics['pose_trans_loss_x'] + best_metrics['pose_trans_loss_y'],
                best_metrics['pose_rot_loss'])
            print('best epoch: {} | total_loss: {:.5f} | {}'.format(
                best_epoch['total_loss'],
                best_metrics['total_loss'],
                loss_str))

    writer.export_scalars_to_json('{}{}/writer_scalars.json'.format(args.ckp_dir, args.exp_name))
    writer.close()


def compute_loss(args, pred_rel_poses, y_rel_poses):
    pose_trans_loss_x = F.mse_loss(pred_rel_poses[:, :, :1] * args.translation_weight,
                                   y_rel_poses[:, :, :1] * args.translation_weight,
                                   reduction='none').sum(dim=2).mean(dim=(0, 1))
    pose_trans_loss_y = F.mse_loss(pred_rel_poses[:, :, 1:2] * args.translation_weight,
                                   y_rel_poses[:, :, 1:2] * args.translation_weight,
                                   reduction='none').sum(dim=2).mean(dim=(0, 1))
    pose_rot_loss = F.mse_loss(pred_rel_poses[:, :, -1:] * args.rotation_weight,
                               y_rel_poses[:, :, -1:] * args.rotation_weight,
                               reduction='none').sum(dim=2).mean(dim=(0, 1))
    total_loss = pose_trans_loss_x + pose_trans_loss_y + pose_rot_loss

    return pose_rot_loss, pose_trans_loss_x, pose_trans_loss_y, total_loss


def main():
    param = Param()
    args = param.get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train(args)
        

if __name__ == '__main__':
    start_time = timer()
    main()
    running_time = timer() - start_time
    print('==============================5==========')
    print('total running time: {:.0f}h:{:2.0f}m:{:2.0f}s'.format(running_time // 3600,
                                                                 (running_time % 3600) // 60,
                                                                 (running_time % 60)))
    print('========================================')



