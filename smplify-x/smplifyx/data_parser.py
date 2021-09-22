# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp

import json
import pickle

from collections import namedtuple

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset


from utils import smpl_to_openpose

Keypoints = namedtuple('Keypoints',
                       ['keypoints', 'gender_gt', 'gender_pd'])

Keypoints.__new__.__defaults__ = (None,) * len(Keypoints._fields)


def create_dataset(dataset='openpose', data_folder='data', **kwargs):
    if dataset.lower() == 'openpose':
        return OpenPose(data_folder, **kwargs)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


def read_keypoints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=False):
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    keypoints = []

    gender_pd = []
    gender_gt = []
    for idx, person_data in enumerate(data['people']):
        body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                  dtype=np.float32)
        body_keypoints = body_keypoints.reshape([-1, 3])
        if use_hands:
            left_hand_keyp = np.array(
                person_data['hand_left_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])
            right_hand_keyp = np.array(
                person_data['hand_right_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])

            body_keypoints = np.concatenate(
                [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)
        if use_face:
            # TODO: Make parameters, 17 is the offset for the eye brows,
            # etc. 51 is the total number of FLAME compatible landmarks
            face_keypoints = np.array(
                person_data['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]

            contour_keyps = np.array(
                [], dtype=body_keypoints.dtype).reshape(0, 3)
            if use_face_contour:
                contour_keyps = np.array(
                    person_data['face_keypoints_2d'],
                    dtype=np.float32).reshape([-1, 3])[:17, :]

            body_keypoints = np.concatenate(
                [body_keypoints, face_keypoints, contour_keyps], axis=0)

        if 'gender_pd' in person_data:
            gender_pd.append(person_data['gender_pd'])
        if 'gender_gt' in person_data:
            gender_gt.append(person_data['gender_gt'])

        keypoints.append(body_keypoints)

    return Keypoints(keypoints=keypoints, gender_pd=gender_pd,
                     gender_gt=gender_gt)



def get_smpl_params(mocap):
    betas = mocap['pred_output_list'][0]['pred_betas']
    body_pose = mocap['pred_output_list'][0]['pred_body_pose']
    pred_camera = mocap['pred_output_list'][0]['pred_camera']
    bbox_scale_ratio = mocap['pred_output_list'][0]['bbox_scale_ratio']
    bbox_top_left = mocap['pred_output_list'][0]['bbox_top_left']

    body_bbox = mocap['body_bbox_list'][0]
    left_hand_bbox = mocap['hand_bbox_list'][0]['left_hand']
    right_hand_bbox = mocap['hand_bbox_list'][0]['right_hand']

    left_hand_pose = mocap['pred_output_list'][0]['pred_left_hand_pose']
    right_hand_pose = mocap['pred_output_list'][0]['pred_right_hand_pose']

    return betas, body_pose, pred_camera, bbox_scale_ratio, bbox_top_left, body_bbox, left_hand_pose, right_hand_pose



def estimate_camera_translation(pred_camera, bbox_scale_ratio, body_bbox, W, H):
    # camera_translation = np.array([pred_camera[1], pred_camera[2],
    #     2*5000/(224*pred_camera[0] + 1e-9)])
    # vertices += camera_translation
    # vertices[:,:2] /= bbox_scale_ratio
    # min_vert = np.min(vertices, axis=0)
    # max_vert = np.max(vertices, axis=0)
    # smpl_bbox_dim = max_vert - min_vert
    # W = 1080
    # H = 1080
    z_scale = 2*5000*bbox_scale_ratio/(224*pred_camera[0] + 1e-9)
    # print("Scale ratios ", smpl_bbox_dim[0]/body_bbox[2],
    #     smpl_bbox_dim[1]/body_bbox[3], z_scale / 5000.0)
    # scale_ratio = smpl_bbox_dim[1]/body_bbox[3] 
    # scale_ratio = smpl_bbox_dim[0]/body_bbox[2]
    scale_ratio = z_scale  / 5000.0
    offset_x = -1*(W/2.0 - (body_bbox[0] + body_bbox[2]/2.0))*scale_ratio
    offset_y = -1*(H/2.0 - (body_bbox[1] + body_bbox[3]/2.0))*scale_ratio
    # vertices[:,0] += offset_x
    # vertices[:,1] += offset_y

    offset_x += pred_camera[1] 
    offset_y += pred_camera[2]
    camera_translation = np.array([offset_x, offset_y, z_scale])

    return camera_translation

def get_total_folders(path):
    i = 0
    for x in os.listdir(path):
        if(".DS_Store" in x):
            continue
        i += 1
        # print("List x ", x)
    return i

def get_smplx_list(data_root):
    base_path = os.path.join(data_root, "frankmocap")
    mocap_path = os.path.join(base_path, "mocap")
    bbox_path = os.path.join(base_path, "bbox")

    num_frames = get_total_folders(mocap_path)
    
    body_pose_list = np.zeros((num_frames, 72))
    beta_list = np.zeros((num_frames, 10))
    cam_trans = np.zeros((num_frames, 3))
    lft_hnd_list = np.zeros((num_frames, 45))
    rgt_hnd_list = np.zeros((num_frames, 45))
    i = 0
    for x in range(num_frames):
        if(".DS_Store" in [x]):
            continue
        bbox = json.load(open(os.path.join(bbox_path, 
            str(x)+"_bbox.json"), "r"))
        mocap = pickle.load(open(os.path.join(mocap_path, 
            str(x)+"_prediction_result.pkl"), "rb"))
        
        # print("Bounding box", bbox['body_bbox_list'],
        #   bbox['hand_bbox_list'][0]['left_hand'], 
        #   bbox['hand_bbox_list'][0]['right_hand'])
        # print("Mocap ", mocap)
        betas, body_pose, pred_camera, bbox_scale_ratio, \
        bbox_top_left, body_bbox, left_hand_pose, right_hand_pose = \
            get_smpl_params(mocap)

        beta_list[x] = betas
        body_pose_list[x] = body_pose
        lft_hnd_list[x] = left_hand_pose
        rgt_hnd_list[x] = right_hand_pose
        cam_trans[x] = estimate_camera_translation(pred_camera, 
            bbox_scale_ratio, body_bbox, W=1080, H=1080)
        # print("Shape ", body_pose.shape)

        i += 1

    # print(np.mean(beta_list, axis=0), np.std(beta_list, axis=0))
    # print(np.mean(cam_trans, axis=0), np.std(cam_trans, axis=0))
    # print(np.mean(body_pose_list, axis=0), np.std(body_pose_list, axis=0))
    return beta_list, cam_trans, body_pose_list, lft_hnd_list, rgt_hnd_list

# List of data required:
# Run frankmocap - to get init camera parameters and betas and pose params
# Run openpose - 2D key points
# Run visible hands and legs and face regions - Human parser
class OpenPose(Dataset):

    NUM_BODY_JOINTS = 25
    NUM_HAND_JOINTS = 20

    def __init__(self, data_folder, img_folder='images',
                 keyp_folder='keypoints',
                 use_hands=False,
                 use_face=False,
                 dtype=torch.float32,
                 model_type='smplx',
                 joints_to_ign=None,
                 use_face_contour=False,
                 openpose_format='coco25',
                 **kwargs):
        super(OpenPose, self).__init__()

        self.use_hands = use_hands
        self.use_face = use_face
        self.model_type = model_type
        self.dtype = dtype
        self.joints_to_ign = joints_to_ign
        self.use_face_contour = use_face_contour

        self.openpose_format = openpose_format

        self.num_joints = (self.NUM_BODY_JOINTS +
                           2 * self.NUM_HAND_JOINTS * use_hands)

        self.img_folder = osp.join(data_folder, img_folder)
        self.keyp_folder = osp.join(data_folder, keyp_folder)

        self.total_files = 10#get_total_folders(self.img_folder)
        self.idx_scale = 20

        # self.img_paths = [osp.join(self.img_folder, img_fn)
        #                   for img_fn in os.listdir(self.img_folder)
        #                   if img_fn.endswith('.png') or
        #                   img_fn.endswith('.jpg') and
        #                   not img_fn.startswith('.')]
        # self.img_paths = sorted(self.img_paths)
        # print("Image paths ", self.img_paths[:20])
        self.cnt = 0
        beta_list, cam_trans, body_pose_list, lft_hnd_list, rgt_hnd_list = \
            get_smplx_list(data_folder)
        self.beta_list = torch.tensor(beta_list, dtype=self.dtype)
        self.cam_trans = torch.tensor(cam_trans, dtype=self.dtype)
        self.body_pose_list = torch.tensor(body_pose_list, dtype=self.dtype)
        self.lft_hnd_list = torch.tensor(lft_hnd_list, dtype=self.dtype)
        self.rgt_hnd_list = torch.tensor(rgt_hnd_list, dtype=self.dtype)
        # self.mean_beta = torch.tensor(np.mean(beta_list, axis=0), dtype=self.dtype)
        self.mean_beta = self.update_mean()

    def get_model2data(self):
        return smpl_to_openpose(self.model_type, use_hands=self.use_hands,
                                use_face=self.use_face,
                                use_face_contour=self.use_face_contour,
                                openpose_format=self.openpose_format)

    def get_left_shoulder(self):
        return 2

    def get_right_shoulder(self):
        return 5

    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(self.num_joints + 2 * self.use_hands +
                                self.use_face * 51 +
                                17 * self.use_face_contour,
                                dtype=np.float32)

        # Neck, Left and right hip
        # These joints are ignored because SMPL has no neck joint and the
        # annotation of the hips is ambiguous.
        if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
            optim_weights[self.joints_to_ign] = 0.
        return torch.tensor(optim_weights, dtype=self.dtype)

    def __len__(self):
        return self.total_files #len(self.img_paths)

    def __getitem__(self, idx):
        idx = self.idx_scale*idx
        img_path = os.path.join(self.img_folder, str(idx)+".jpg") #self.img_paths[idx]
        # betas, body_pose, cam_trans = self.read_init_param(idx)
        output_dict = self.read_item(img_path)
        output_dict["betas"] = self.beta_list[idx]
        output_dict["cam_trans"] = self.cam_trans[idx]
        output_dict["body_pose"] = self.body_pose_list[idx]
        output_dict["left_hand_pose"] = self.lft_hnd_list[idx]
        output_dict["right_hand_pose"] = self.rgt_hnd_list[idx]
        output_dict["mean_beta"] = self.mean_beta
        # print("Output data ", output_dict)
        return output_dict

    def update_new_idx(self, idx, cam_trans, body_pose, left_hand_pose,
        right_hand_pose):
        idx = self.idx_scale*idx
        self.cam_trans[idx] = torch.tensor(cam_trans, dtype=self.dtype)
        self.body_pose_list[idx, :66] = torch.tensor(body_pose, dtype=self.dtype)
        self.lft_hnd_list[idx] = torch.tensor(left_hand_pose, dtype=self.dtype)
        self.rgt_hnd_list[idx] = torch.tensor(right_hand_pose, dtype=self.dtype)
        return

    def update_beta(self, beta):
        self.mean_beta = beta
        return

    def update_mean(self):
        mean_beta = torch.zeros((1, 10))
        for i in range(self.total_files):
            idx = i*self.idx_scale
            mean_beta += self.beta_list[idx]

        self.mean_beta = mean_beta/self.total_files
        # print("The mean beta ", self.mean_beta)
        return self.mean_beta

    def read_item(self, img_path):
        img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0
        img_fn = osp.split(img_path)[1]
        img_fn, _ = osp.splitext(osp.split(img_path)[1])

        keypoint_fn = osp.join(self.keyp_folder,
                               img_fn + '_keypoints.json')
        keyp_tuple = read_keypoints(keypoint_fn, use_hands=self.use_hands,
                                    use_face=self.use_face,
                                    use_face_contour=self.use_face_contour)

        if len(keyp_tuple.keypoints) < 1:
            return {}

        

        keypoints = np.stack(keyp_tuple.keypoints)

        output_dict = {'fn': img_fn,
                       'img_path': img_path,
                       'keypoints': keypoints, 'img': img}
        if keyp_tuple.gender_gt is not None:
            if len(keyp_tuple.gender_gt) > 0:
                output_dict['gender_gt'] = keyp_tuple.gender_gt
        if keyp_tuple.gender_pd is not None:
            if len(keyp_tuple.gender_pd) > 0:
                output_dict['gender_pd'] = keyp_tuple.gender_pd
        return output_dict

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.cnt >= self.total_files: #len(self.img_paths):
            raise StopIteration

        # img_path = self.img_paths[self.cnt]
        inp_idx = self.cnt*self.idx_scale
        img_path = os.path.join(self.img_folder, str(inp_idx)+".jpg") #self.img_paths[idx]

        output_dict = self.read_item(img_path)
        output_dict["betas"] = self.beta_list[inp_idx]
        output_dict["cam_trans"] = self.cam_trans[inp_idx]
        output_dict["body_pose"] = self.body_pose_list[inp_idx]
        output_dict["mean_beta"] = self.mean_beta
        output_dict["left_hand_pose"] = self.lft_hnd_list[inp_idx]
        output_dict["right_hand_pose"] = self.rgt_hnd_list[inp_idx]
        self.cnt += 1

        return output_dict
