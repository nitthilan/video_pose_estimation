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

import pose_fitting
# from human_body_prior.tools.model_loader import load_vposer
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.tools.model_loader import load_model
import smplx
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from camera import create_camera

import common_functions as cf


if mp.get_start_method(allow_none=True) != 'forkserver':
    mp.set_start_method('forkserver', force=True)


def refine_pose(dataset_obj, result_folder, model_params,
        camera, joint_weights, dtype,
        shape_prior, expr_prior, body_pose_prior, left_hand_prior,
        right_hand_prior, jaw_prior, angle_prior, **args):

    max_persons = 1
    input_gender = "neutral"
    use_cuda = True



    train_dataloader = DataLoader(dataset_obj, batch_size=1, shuffle=False)

    results_list = []
    body_pose_list = np.zeros((10,66))
    betas_list = np.zeros((10,10))
    camera_trans_list = np.zeros((10,3))

    org_body_pose_list = np.zeros((10,72))
    org_betas_list = np.zeros((10,10))
    org_camera_trans_list = np.zeros((10,3))

    pool = mp.Pool(processes=5)

    result_async_list = []
    def accumulate_result(result):
        # result_async_list.append(result)
        return

    def handle_error(exception):
        print("Exception ", exception)
        return

    for idx, data in enumerate(train_dataloader):
        # print("The data keys ", data.keys())

        fn = data['fn'][0]
        keypoints = data['keypoints'][0]
        print('Processing: {}'.format(data['img_path'][0]))

        curr_result_folder = osp.join(result_folder, fn)
        if not osp.exists(curr_result_folder):
            os.makedirs(curr_result_folder)

        for person_id in range(keypoints.shape[0]):
            if person_id >= max_persons and max_persons > 0:
                continue

            curr_result_fn = osp.join(curr_result_folder,
                                      '{:03d}.pkl'.format(person_id))
            # if gender_lbl_type != 'none':
            #     if gender_lbl_type == 'pd' and 'gender_pd' in data:
            #         gender = data['gender_pd'][person_id]
            #     if gender_lbl_type == 'gt' and 'gender_gt' in data:
            #         gender = data['gender_gt'][person_id]
            # else:
            gender = input_gender
            # results = fit_single_frame(data, keypoints[[person_id]],
            #                  gender=gender,
            #                  model_params=model_params,
            #                  camera=camera,
            #                  joint_weights=joint_weights,
            #                  dtype=dtype,
            #                  result_folder=curr_result_folder,
            #                  result_fn=curr_result_fn,
            #                  shape_prior=shape_prior,
            #                  expr_prior=expr_prior,
            #                  body_pose_prior=body_pose_prior,
            #                  left_hand_prior=left_hand_prior,
            #                  right_hand_prior=right_hand_prior,
            #                  jaw_prior=jaw_prior,
            #                  angle_prior=angle_prior,
            #                  **args)

            result_async = pool.apply_async(fit_single_frame, 
                            (data, keypoints[[person_id]],),
                             dict(gender=gender,
                             model_params=model_params,
                             joint_weights=joint_weights.clone(),
                             dtype=dtype,
                             result_folder=curr_result_folder,
                             result_fn=curr_result_fn,
                             **args),
                             callback = accumulate_result,
                             error_callback=handle_error)
            result_async_list.append(result_async)

    pool.close()
    pool.join()


    for idx, data in enumerate(train_dataloader):
        results = result_async_list[idx].get()
        results_list.append(results)
        # print("Shape ", results[1]['result'].keys())
        # print("Shape ", results[1]['result']['camera_rotation'].shape,
        #     results[1]['result']['camera_translation'].shape,
        #     results[1]['result']['betas'].shape,
        #     results[1]['result']['body_pose'].shape,
        #     results[1]['result']['left_hand_pose'].shape)
        betas_list[idx,:] = results[1]['result']['betas']
        body_pose_list[idx, 3:66] = results[1]['result']['body_pose']
        body_pose_list[idx, :3] = results[1]['result']['global_orient']
        camera_trans_list[idx, :] = results[1]['result']['camera_translation']

        org_betas_list[idx, :] = data['betas']
        org_camera_trans_list[idx, :] = data['cam_trans']
        org_body_pose_list[idx, :] = data['body_pose']

        dataset_obj.update_new_idx(idx, camera_trans_list[idx], 
            body_pose_list[idx], 
            results[1]['result']['left_hand_pose'],
            results[1]['result']['right_hand_pose'])

    print("Mean betas ", np.mean(betas_list, axis=0),
        np.std(betas_list, axis=0))

    print("Org Mean betas ", np.mean(org_betas_list, axis=0),
        np.std(org_betas_list, axis=0))
    print("Cam Trans ", org_camera_trans_list, camera_trans_list)
    return


def fit_single_frame(data,
                     keypoints,
                     gender,
                     model_params,
                     joint_weights,
                     result_fn='out.pkl',
                     loss_type='smplify',
                     use_cuda=True,
                     init_joints_idxs=(9, 12, 2, 5),
                     use_face=True,
                     use_hands=True,
                     data_weights=None,
                     body_pose_prior_weights=None,
                     hand_pose_prior_weights=None,
                     jaw_pose_prior_weights=None,
                     shape_weights=None,
                     expr_weights=None,
                     hand_joints_weights=None,
                     face_joints_weights=None,
                     depth_loss_weight=1e2,
                     interpenetration=True,
                     coll_loss_weights=None,
                     df_cone_height=0.5,
                     penalize_outside=True,
                     max_collisions=8,
                     point2plane=False,
                     part_segm_fn='',
                     focal_length=5000.,
                     side_view_thsh=25.,
                     rho=100,
                     vposer_latent_dim=32,
                     vposer_ckpt='',
                     use_joints_conf=False,
                     interactive=True,
                     visualize=False,
                     save_meshes=True,
                     batch_size=1,
                     dtype=torch.float32,
                     ign_part_pairs=None,
                     left_shoulder_idx=2,
                     right_shoulder_idx=5,
                     **kwargs):
    assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    camera = cf.get_camera(kwargs.get('focal_length'), use_cuda, dtype, kwargs)

    body_model = cf.get_model(use_cuda, model_params, gender)

    body_pose_prior, jaw_prior, expr_prior, left_hand_prior, \
        right_hand_prior, shape_prior, angle_prior = \
            cf.get_priors(kwargs, dtype)

    opt_weights = cf.get_weights(data_weights, body_pose_prior_weights, 
                hand_pose_prior_weights, shape_weights, jaw_pose_prior_weights, 
                expr_weights, face_joints_weights, hand_joints_weights, 
                use_hands, use_face, device, dtype)

    betas = torch.unsqueeze(data['mean_beta'][0], 0).float().to(device=device)
    body_pose = torch.unsqueeze(data['body_pose'][0], 0).float().to(device=device)
    # print("The input pose info ", body_pose.shape)

    keypoint_data = keypoints
    gt_joints = keypoint_data[:, :, :2]
    if use_joints_conf:
        joints_conf = keypoint_data[:, :, 2].reshape(1, -1)

    # Transfer the data to the correct device
    gt_joints = gt_joints.to(device=device, dtype=dtype)
    if use_joints_conf:
        joints_conf = joints_conf.to(device=device, dtype=dtype)


    # The indices of the joints used for the initialization of the camera
    loss = pose_fitting.create_loss(loss_type=loss_type,
                               joint_weights=joint_weights,
                               rho=rho,
                               use_joints_conf=use_joints_conf,
                               use_face=use_face, use_hands=use_hands,
                               body_pose_prior=body_pose_prior,
                               shape_prior=shape_prior,
                               angle_prior=angle_prior,
                               expr_prior=expr_prior,
                               left_hand_prior=left_hand_prior,
                               right_hand_prior=right_hand_prior,
                               jaw_prior=jaw_prior,
                               dtype=dtype,
                               **kwargs)
    loss = loss.to(device=device)

    monitor = pose_fitting.FittingMonitor(batch_size=batch_size, visualize=visualize, **kwargs)

    # if(model_type == "smpl"):
    #     wrist_pose = torch.zeros([body_pose.shape[0], 6],
    #                          dtype=body_pose.dtype,
    #                          device=device)
    #     body_pose = torch.cat([body_pose, wrist_pose], dim=1)
    #     est_params['body_pose'] = body_pose[:, 3:72]
    # else:
    est_params = {}
    est_params['betas'] = betas
    est_params['body_pose'] = body_pose[:, 3:66]
    est_params['global_orient'] = body_pose[:, :3]
    est_params['left_hand_pose'] = data['left_hand_pose'][0]
    est_params['right_hand_pose'] = data['right_hand_pose'][0]
    body_model.reset_params(**est_params)

    img = data['img'][0]

    H, W, _ = img.shape

    data_weight = 1000 / H
    
    # Update the value of the translation of the camera as well as
    # the image center.
    with torch.no_grad():
        camera.translation[:] = data['cam_trans'][0].view_as(camera.translation) #torch.tensor(data['cam_trans'], dtype=dtype) #init_t.view_as(camera.translation)
        camera.center[:] = torch.tensor([W, H], dtype=dtype) * 0.5
    # print("Camera params ", camera.focal_length_x, camera.focal_length_y)
    # Re-enable gradient calculation for the camera translation
    camera.translation.requires_grad = True
    # store here the final error for both orientations,
    # and pick the orientation resulting in the lowest error
    results = []

    # Step 2: Optimize the full model
    final_loss_val = 0
    opt_start = time.time()

    for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):

        # body_params = list(body_model.parameters())
        body_params = list(body_model.named_parameters())

        # print("Body params ", dir(body_model))
        # for params in body_model.named_parameters():
        #     print(params)

        # final_params = list(
        #     filter(lambda x: x.requires_grad , body_params))

        final_params = []
        for k, v in body_model.named_parameters():
            if k != 'betas':
            # if k != 'body_pose' and k != 'global_orient':
                final_params.append(v)
                # print("Final params ", k)
        final_params.append(camera.translation)

        body_optimizer, body_create_graph = optim_factory.create_optimizer(
            final_params,
            **kwargs)
        body_optimizer.zero_grad()

        curr_weights['data_weight'] = data_weight
        curr_weights['bending_prior_weight'] = (
            3.17 * curr_weights['body_pose_weight'])
        if use_hands:
            joint_weights[:, 25:67] = curr_weights['hand_weight']
        if use_face:
            joint_weights[:, 67:] = curr_weights['face_weight']
        loss.reset_loss_weights(curr_weights)

        # print("Current weights ", curr_weights.keys(),
        #     joint_weights.shape)


        closure = monitor.create_fitting_closure(
            body_optimizer, body_model,
            camera=camera, gt_joints=gt_joints,
            joints_conf=joints_conf,
            joint_weights=joint_weights,
            loss=loss, create_graph=body_create_graph,
            return_verts=True, return_full_pose=True)

        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            stage_start = time.time()
        final_loss_val = monitor.run_fitting(
            body_optimizer,
            closure, final_params,
            body_model)

        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - stage_start
            if interactive:
                tqdm.write('Stage {:03d} done after {:.4f} seconds. Loss {}'.format(
                    opt_idx, elapsed, final_loss_val))


        # Get the result of the fitting process
        # Store in it the errors list in order to compare multiple
        # orientations, if they exist
        result = {'camera_' + str(key): val.detach().cpu().numpy()
                  for key, val in camera.named_parameters()}
        result.update({key: val.detach().cpu().numpy()
                       for key, val in body_model.named_parameters()})

        results.append({'loss': final_loss_val,
                        'result': result})

    return results
