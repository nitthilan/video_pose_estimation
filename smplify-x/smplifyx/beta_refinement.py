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

def refine_beta(dataset_obj, result_folder, model_params,
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

    final_params = []
    for k, v in body_model.named_parameters():
        if k == 'betas':
            final_params.append(v)
            
    body_optimizer, body_create_graph = optim_factory.create_optimizer(
        final_params, **args)

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
    curr_weights, joint_weights = init_weights(joint_weights, data_weight,
        body_pose_prior_weights[1], shape_weights[1], 
        face_joints_weights[1], hand_joints_weights[1],
        use_hands, use_face)
    loss.reset_loss_weights(curr_weights)




    with beta_fitting.FittingMonitor(**args) as monitor:


        # store here the final error for both orientations,
        # and pick the orientation resulting in the lowest error
        results = []

        # Step 2: Optimize the full model
        final_loss_val = 0
        opt_start = time.time()
        body_optimizer.zero_grad()

        closure = monitor.create_fitting_closure(
            body_optimizer, body_model,
            camera=camera,
            joint_weights=joint_weights,
            loss=loss, create_graph=body_create_graph,
            dataset_obj=dataset_obj,
            return_verts=True, return_full_pose=True,
            dtype=dtype,
            use_joints_conf=use_joints_conf)

        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            stage_start = time.time()
        final_loss_val = monitor.run_fitting(body_optimizer, 
            closure, final_params, body_model)

        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - stage_start
            if interactive:
                tqdm.write('done after {:.4f} seconds. Loss {}'.format(
                    elapsed, final_loss_val))


        # Get the result of the fitting process
        # Store in it the errors list in order to compare multiple
        # orientations, if they exist
        result = {'camera_' + str(key): val.detach().cpu().numpy()
                  for key, val in camera.named_parameters()}
        result.update({key: val.detach().cpu().numpy()
                       for key, val in body_model.named_parameters()})

        results.append({'loss': final_loss_val,
                        'result': result})
    return

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
