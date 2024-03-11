import csv
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import os
import glob
from pathlib import Path
import numpy as np
import flownet_utils.frame_utils as frame_utils
from flownet_utils.frame_utils import StaticCenterCrop
import PIL
from PIL import Image
import torch


def read_mit_pose(base_dir, sequence_gt):
    """
    -> base_dir: data/kitti/odometry/dataset
    -> read the ground-truth pose for each image
    -> list of poses (np.array: (12,))
    """
    poses = []
    with open('{}/ground_truth/{}.gt.laser.poses'.format(base_dir, sequence_gt), mode='r', encoding='utf-8-sig') as f:
        gt_reader = csv.reader(f, delimiter=',')
        for row in gt_reader:
            poses.append(row)

        dataframe = pd.DataFrame(poses, columns=['timestamp', 'x', 'y', 'theta'])
        dataframe['timestamp'] = dataframe['timestamp'].astype('int64')
        dataframe['x'] = dataframe['x'].astype('float64')
        dataframe['y'] = dataframe['y'].astype('float64')
        dataframe['theta'] = dataframe['theta'].astype('float64')

    return dataframe

def read_mit_img(base_dir, sequence, start, stop):
    """
    -> base_dir: data/mit
    -> Return: list of image filenames
    """
    imgs = []
    timestamps = []
    path_img = '{}/sequences/{}/rgb/'.format(base_dir, sequence)
    temp = sorted(glob.glob(path_img, '*.{}'.format('.jpg')))
    for i in range(len(temp)):
        frame_time = int(float(Path(temp[i]).stem))

        if start <= frame_time <= stop:
            img_label = '{}-{}.jpg'.format(sequence, Path(temp[i]).stem)
            imgs.append([frame_time, img_label])
            timestamps.append(frame_time)

    timestamps = np.array(timestamps, dtype=np.uint64)
    return timestamps, imgs

def read_mit_imu(base_dir, sequence, start, stop, timestamps):
    """
    -> base_dir: data/kitti/odometry/dataset
    -> Return: list of [imu_label (00-000000), np.array(11, 6)]
        -> the inner list is the imu data between two images (len: 11,12,14 -> trim to 11)
    -> format of raw imu data ([0] ~ [29])
        * [11] ax: acceleration in x (m/s^2)
        * [12] ay: acceleration in y (m/s^2)
        * [13] az: acceleration in z (m/s^2)
        * [17] wx: angular rate around x (rad/s)
        * [18] wy: angular rate around y (rad/s)
        * [19] wz: angular rate around z (rad/s)
    """
    # wx, wy, wz, ax, ay, az
    path_imu = '{}/sequences/{}/imu/'.format(base_dir, sequence)
    file = os.path.join(path_imu, "imu.txt")
    data = np.loadtxt(file)
    data = np.delete(data, 
                     np.where((data[:, 0].astype(np.uint64) <= start) | (data[:, 0].astype(np.uint64) >= stop)), axis=0)
    imus = []

    for i in range(timestamps.shape[0] - 1):
        frame2_time = timestamps[i + 1]
        data_imu = np.delete(data, np.where(data[:, 0].astype(np.uint64) > frame2_time), axis=0)
        imus.append([frame2_time, np.concatenate((data_imu[-3:, 8:11], data_imu[-3:, 5:8]), axis=1)])

    return imus

def get_pose_by_timestamps(poses, timestamps):
    new_poses = []
    for i in range(timestamps.shape[0] - 1):
        x = np.interp(timestamps[i+1], poses['timestamp'], poses['x'])
        y = np.interp(timestamps[i+1], poses['timestamp'], poses['y'])
        theta = np.interp(timestamps[i+1], poses['timestamp'], poses['theta'])
        new_poses.append([timestamps[i+1], np.array([x, y, 0, 0, 0, theta])])

    return new_poses

def get_mit_depthpair(last_img, curr_img, base_dir, render_size=None):
    """
    (1) last_img: '00-000000'
    (2) curr_img: '00-000001'
    """
    # last_seq = last_img.split('-')[0]
    # last_idx = int(last_img.split('-')[1])
    # curr_seq = curr_img.split('-')[0]
    # curr_idx = int(curr_img.split('-')[1])
    # assert last_seq == curr_seq
    # assert int(last_idx) + 1 == int(curr_idx)
    # last_depth_path = '{}/depths/{}/{:010d}.png'.format(base_dir, last_seq, last_idx)
    # curr_depth_path = '{}/depths/{}/{:010d}.png'.format(base_dir, curr_seq, curr_idx)
    
    # assert render_size is not None
    # depth1 = Image.open(last_depth_path)
    # depth2 = Image.open(curr_depth_path)
    # depth1 = np.array(depth1, dtype=np.float32) / 256.0
    # depth2 = np.array(depth2, dtype=np.float32) / 256.0
    # depths = [depth1, depth2]
    # depth_size = depth1.shape[:2]
    # cropper = StaticCenterCrop(depth_size, render_size)
    # depths = list(map(cropper, depths))

    # # [2, 320, 1216] (No need to transpose)
    # depths = np.array(depths)
    # depths = torch.from_numpy(depths)
    # return depths
    raise NotImplementedError()

def get_mit_imgpair(last_img, curr_img, base_dir, img_transforms=None, render_size=None):
    """
    (1) last_img: '00-000000'
    (2) curr_img: '00-000001'
    """
    if img_transforms is None and render_size is None: 
        raise ValueError('one and only on of img_transforms and render_size should be given')
    if img_transforms is not None and render_size is not None: 
        raise ValueError('one and only on of img_transforms and render_size should be given')
    
    last_seq = last_img.split('-')[0]
    last_filename = last_img.split('-')[1]
    curr_seq = curr_img.split('-')[0]
    curr_filename = curr_img.split('-')[1]
    assert last_seq == curr_seq
    last_img_path = '{}/sequences/{}/rgb/{}'.format(base_dir, last_seq, last_filename)
    curr_img_path = '{}/sequences/{}/rgb/{}'.format(base_dir, curr_seq, curr_filename)

    if img_transforms is None and render_size is None: 
        raise ValueError('one and only on of img_transforms and render_size should be given')
    if img_transforms is not None and render_size is not None: 
        raise ValueError('one and only on of img_transforms and render_size should be given')

    if img_transforms is None:
        # use FlowNet2/C/S
        assert render_size is not None
        img1 = frame_utils.read_gen(last_img_path)
        img2 = frame_utils.read_gen(curr_img_path)
        images = [img1, img2]
        image_size = img1.shape[:2]
        cropper = StaticCenterCrop(image_size, render_size)
        images = list(map(cropper, images))
        images = np.array(images).transpose(3,0,1,2)

        # tmp_time = timer()
        # images = images.astype(np.float32)
        # tout = timer() - tmp_time
        # if tout > 1: print('np_float32: {} s'.format(tout))

        images = torch.from_numpy(images)
        return images
    else:
        # if train_img_from_scrach: ToTensor will transform (H,W,C) PIL image in [0,255] to (C,H,W) in [0.0,1.0]
        r_last_img = img_transforms(PIL.Image.open(last_img_path)) # [3, 192, 640] for kitti
        r_curr_img = img_transforms(PIL.Image.open(curr_img_path)) # [3, 192, 640] for kitti
        return torch.stack((r_last_img, r_curr_img), dim=1).type(torch.FloatTensor)