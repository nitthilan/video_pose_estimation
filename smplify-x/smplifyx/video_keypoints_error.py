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


import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys
import os
import os.path as osp

import numpy as np
import torch

from tqdm import tqdm

from collections import defaultdict

import cv2
import PIL.Image as pil_img

from optimizers import optim_factory

import beta_fitting
# from human_body_prior.tools.model_loader import load_vposer
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.tools.model_loader import load_model
import smplx
from torch.utils.data import DataLoader
import utils

def get_model(use_cuda, model_params, input_gender):
    male_model = smplx.create(gender='male', **model_params)
    # SMPL-H has no gender-neutral model
    neutral_model = smplx.create(gender='neutral', **model_params)
    female_model = smplx.create(gender='female', **model_params)

    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')

        female_model = female_model.to(device=device)
        male_model = male_model.to(device=device)
        neutral_model = neutral_model.to(device=device)

    gender = input_gender

    if gender == 'neutral':
        body_model = neutral_model
    elif gender == 'female':
        body_model = female_model
    elif gender == 'male':
        body_model = male_model


    return body_model

def video_keypoints_error(dataset_obj, result_folder, model_params,
        camera, joint_weights, dtype,
        shape_prior, expr_prior, body_pose_prior, left_hand_prior,
        right_hand_prior, jaw_prior, angle_prior, 
        body_pose_prior_weights=None, shape_weights=None, 
        hand_joints_weights=None, face_joints_weights=None,
        use_hands=None, use_face= None, interactive=True,
        **args):

    max_persons = 1
    input_gender = "neutral"
    use_cuda = True
    person_id = 0
    max_persons = 1
    rho=100
    loss_type = 'smplify'
    use_joints_conf = True
    body_model = get_model(use_cuda, model_params, input_gender)
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    # assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'
    # if person_id >= max_persons and max_persons > 0:
    #     continue
    sample_data=next(iter(dataset_obj))

    betas = torch.unsqueeze(sample_data['mean_beta'], 0).float().to(device=device)
    est_params = {}
    est_params['betas'] = betas
    body_model.reset_params(**est_params)

    # The indices of the joints used for the initialization of the camera
    loss = beta_fitting.create_loss(loss_type=loss_type, rho=rho,
                               use_joints_conf=use_joints_conf,
                               body_pose_prior=body_pose_prior,
                               shape_prior=shape_prior,
                               dtype=dtype)
    loss = loss.to(device=device)



    body_pose_prior_weights, shape_weights, \
        hand_joints_weights, face_joints_weights = \
        get_loss_weights(body_pose_prior_weights,
            shape_weights, hand_joints_weights, face_joints_weights,
            use_hands, use_face)

    img = torch.tensor(sample_data['img'], dtype=dtype)
    H, W, _ = img.shape
    camera.center[:] = torch.tensor([W, H], dtype=dtype) * 0.5

    data_weight = 1000 / H
    # curr_weights, joint_weights = init_weights(joint_weights, data_weight,
    #     body_pose_prior_weights[1], shape_weights[1], 
    #     face_joints_weights[1], hand_joints_weights[1],
    #     use_hands, use_face)
    curr_weights, joint_weights = init_weights(joint_weights, data_weight,
        body_pose_prior_weights[0], shape_weights[0], 
        face_joints_weights[0], hand_joints_weights[0],
        use_hands, use_face)
    loss.reset_loss_weights(curr_weights)

    train_dataloader = DataLoader(dataset_obj, batch_size=1, shuffle=False)

    # torch.autograd.set_detect_anomaly(True)
    person_id = 0
    total_loss = 0.0

    for idx, data in enumerate(train_dataloader):
        # print("The data keys ", data.keys(), data['body_pose'].shape)
        # body_pose = torch.unsqueeze(data['body_pose'], 0).float().to(device=device)
        body_pose = data['body_pose'].float().to(device=device)
        est_params = {}
        est_params['betas'] = betas
        # est_params['betas'] = betas
        est_params['body_pose'] = body_pose[:, 3:66]
        est_params['global_orient'] = body_pose[:, :3]        
        body_model.reset_params(**est_params)


        # Update the value of the translation of the camera as well as
        # the image center.
        with torch.no_grad():
            camera.translation[:] = data['cam_trans'].view_as(camera.translation) #torch.tensor(data['cam_trans'], dtype=dtype) #init_t.view_as(camera.translation)

        # print('Processing: {}'.format(data['img_path']))
        keypoint_data = data['keypoints'][0][[person_id]]
        gt_joints = keypoint_data[:, :, :2]
        gt_joints = gt_joints.to(device=device, dtype=dtype)
        if use_joints_conf:
            joints_conf = keypoint_data[:, :, 2].reshape(1, -1)
            joints_conf = joints_conf.to(device=device, dtype=dtype)

        body_model_output = body_model(return_verts=True,
                                    # body_pose=body_pose[:, 3:66],
                                    # global_orient = body_pose[:, :3],
                                    return_full_pose=True)
        model_loss = loss(body_model_output, camera=camera,
                          gt_joints=gt_joints,
                          joints_conf=joints_conf,
                          joint_weights=joint_weights,
                          **args)
        # model_loss.backward(create_graph=body_create_graph)
        if(model_loss != model_loss):
            print("Error in loss ", idx, est_params, camera.translation, model_loss)
            exit()

        total_loss += model_loss

    print("Average Loss ", idx, total_loss/(idx+1))
        # total_loss = 0.0
        # print("Betas ", idx, body_model.betas)

    return total_loss/(idx+1)

def get_loss_weights(body_pose_prior_weights,
    shape_weights, hand_joints_weights, face_joints_weights,
    use_hands, use_face):

    if body_pose_prior_weights is None:
        body_pose_prior_weights = [4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78]

    if shape_weights is None:
        shape_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
    msg = ('Number of Body pose prior weights = {} does not match the' +
           ' number of Shape prior weights = {}')
    assert (len(shape_weights) ==
            len(body_pose_prior_weights)), msg.format(
                len(shape_weights),
                len(body_pose_prior_weights))

    if use_hands:
        if hand_joints_weights is None:
            hand_joints_weights = [0.0, 0.0, 0.0, 1.0]
            msg = ('Number of Body pose prior weights does not match the' +
                   ' number of hand joint distance weights')
            assert (len(hand_joints_weights) ==
                    len(body_pose_prior_weights)), msg
    if use_face:
        if face_joints_weights is None:
            face_joints_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of face joint distance weights')
        assert (len(face_joints_weights) ==
                len(body_pose_prior_weights)), msg

    return body_pose_prior_weights, shape_weights, hand_joints_weights, \
        face_joints_weights



def init_weights(joint_weights, data_weight, body_pose_weight, 
    shape_weight, face_joints_weights, hand_joints_weights,
    use_hands, use_face):
    curr_weights = {}
    curr_weights['body_pose_weight'] = body_pose_weight
    curr_weights['shape_weight'] = shape_weight
    curr_weights['data_weight'] = data_weight

    if use_hands:
        joint_weights[:, 25:67] = face_joints_weights
    if use_face:
        joint_weights[:, 67:] = hand_joints_weights
    return curr_weights, joint_weights
