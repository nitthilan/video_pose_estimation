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


import common_functions as cf
# from torch import vmap



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
    rho=100
    maxiters = 10
    loss_type = 'smplify'
    use_joints_conf = True
    batch_size = 10
    model_params['batch_size'] = batch_size
    args['batch_size'] = batch_size

    body_model = cf.get_model(use_cuda, model_params, input_gender)
    # batched_body_model = torch.vmap(body_model)
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    # assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'
    # if person_id >= max_persons and max_persons > 0:
    #     continue
    sample_data=next(iter(dataset_obj))

    betas = sample_data['mean_beta'].float().to(device=device)
    est_params = {}
    est_params['betas'] = betas.repeat(batch_size, 1)
    body_model.reset_params(**est_params)

    final_params = []
    for k, v in body_model.named_parameters():
        if k == 'betas':
            final_params.append(v)
            
    body_optimizer, body_create_graph = optim_factory.create_optimizer(
        final_params, optim_type='lbfgsls')

    # The indices of the joints used for the initialization of the camera
    loss = beta_fitting.create_loss(loss_type=loss_type, rho=rho,
                               use_joints_conf=use_joints_conf,
                               body_pose_prior=body_pose_prior,
                               shape_prior=shape_prior,
                               dtype=dtype)
    loss = loss.to(device=device)



    body_pose_prior_weights, shape_weights, \
        hand_joints_weights, face_joints_weights = \
        cf.get_beta_refinement_weights(body_pose_prior_weights,
            shape_weights, hand_joints_weights, face_joints_weights,
            use_hands, use_face)

    camera = cf.get_camera(args.get('focal_length'), use_cuda, dtype, args)

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
        use_hands, use_face, batch_size)
    loss.reset_loss_weights(curr_weights)

    train_dataloader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=False)

    torch.autograd.set_detect_anomaly(True)
    person_id = 0
    total_loss = 0.0
    min_loss = 99999999.0
    min_loss_beta = betas
    for n in range(maxiters):
        for idx, data in enumerate(train_dataloader):
            # print("The data keys ", data.keys(), data['body_pose'].shape)
            # body_pose = torch.unsqueeze(data['body_pose'], 0).float().to(device=device)
            body_pose = data['body_pose'].float().to(device=device)
            # est_params = {}
            # est_params['betas'] = betas
            # # est_params['betas'] = betas
            # est_params['body_pose'] = body_pose[:, 3:66]
            # est_params['global_orient'] = body_pose[:, :3]        
            # body_model.reset_params(**est_params)


            # Update the value of the translation of the camera as well as
            # the image center.
            with torch.no_grad():
                # print("Shape of translation ", camera.translation.shape,
                #     data['cam_trans'].shape)
                camera.translation[:] = data['cam_trans'].view_as(camera.translation) #torch.tensor(data['cam_trans'], dtype=dtype) #init_t.view_as(camera.translation)
            keypoints = data['keypoints']
            # print('Processing: {}'.format(data['img_path']))
            # print("Key point shape ", keypoints.shape)
            keypoints = keypoints[:,person_id,:,:]
            keypoint_data = torch.tensor(keypoints, dtype=dtype)
            gt_joints = keypoint_data[:, :, :2]
            gt_joints = gt_joints.to(device=device, dtype=dtype)
            if use_joints_conf:
                joints_conf = keypoint_data[:, :, 2].reshape(batch_size, -1)
                joints_conf = joints_conf.to(device=device, dtype=dtype)

            def closure():
                body_optimizer.zero_grad()

                body_model_output = body_model(return_verts=True,
                                            body_pose=body_pose[:, 3:66],
                                            global_orient = body_pose[:, :3],
                                            return_full_pose=True)
                model_loss = loss(body_model_output, camera=camera,
                                  gt_joints=gt_joints,
                                  joints_conf=joints_conf,
                                  joint_weights=joint_weights,
                                  **args)
                model_loss.backward(create_graph=body_create_graph)
                return model_loss

            model_loss = body_optimizer.step(closure)

            if torch.isnan(model_loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(model_loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            ftol=2e-09
            gtol=1e-05
            # if n > 0 and prev_loss is not None and ftol > 0:
            #     loss_rel_change = utils.rel_change(prev_loss, model_loss.item())

            #     if loss_rel_change <= ftol:
            #         print("Relative change ", loss_rel_change, ftol)
            #         break

            if all([torch.abs(var.grad.view(-1).max()).item() < gtol
                    for var in final_params if var.grad is not None]):
                print(([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]))
                print("Breaking now ", self.gtol, params[0].grad,
                    params[0])
                break

            prev_loss = model_loss.item()

            total_loss += prev_loss

        total_loss = total_loss/((idx+1)*batch_size)
        if(min_loss >= total_loss):
            min_loss_beta = body_model.betas
            min_loss = total_loss
        print("Loss ", n, idx, total_loss)
        # Update the betas
        est_params = {}
        est_params['betas'] = torch.mean(body_model.betas, dim=0).repeat(batch_size, 1)
        body_model.reset_params(**est_params)

        total_loss = 0.0
        # print("Betas ", idx, body_model.betas)

    dataset_obj.update_beta(torch.mean(min_loss_beta.detach().cpu(), dim=0))

    model_params['batch_size'] = 1
    args['batch_size'] = 1

    return



def init_weights(joint_weights, data_weight, body_pose_weight, 
    shape_weight, face_joints_weights, hand_joints_weights,
    use_hands, use_face, batch_size):
    curr_weights = {}
    curr_weights['body_pose_weight'] = body_pose_weight
    curr_weights['shape_weight'] = shape_weight
    curr_weights['data_weight'] = data_weight

    if use_hands:
        joint_weights[:, 25:67] = face_joints_weights
    if use_face:
        joint_weights[:, 67:] = hand_joints_weights

    joint_weights = joint_weights.repeat(batch_size,1)
    return curr_weights, joint_weights
