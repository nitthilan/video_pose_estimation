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
import pickle


if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)


def fill_gpu(process_queue, num_gpus, num_proc_per_gpu):
    # initialize the queue with the GPU ids
    for gpu_id in range(num_gpus):
        for _ in range(num_proc_per_gpu):
            process_queue.put(gpu_id)
            # print(" Put gpu id ", gpu_id)

    # print("Get id ", process_queue.get())
    return
def refine_pose(dataset_obj, result_folder, model_params,
        camera, joint_weights, dtype,
        shape_prior, expr_prior, body_pose_prior, left_hand_prior,
        right_hand_prior, jaw_prior, angle_prior, **args):

    max_persons = 1
    input_gender = "neutral"
    use_cuda = True
    batch_size = 1
    num_gpus = 4
    num_proc_per_gpu = 5
    output_folder="/nitthilan/data/neuralbody/people_snapshot_public/female-1-casual/shape_pose_refinement/"

    args["batch_size"] = batch_size
    model_params['batch_size'] = batch_size
    joint_weights = joint_weights.repeat(batch_size,1)

    train_dataloader = DataLoader(dataset_obj, batch_size=batch_size, 
        shuffle=False)

    pool = mp.Pool(processes=num_proc_per_gpu*num_gpus)
    process_queue =  mp.Manager().Queue()
    fill_gpu(process_queue, num_gpus, num_proc_per_gpu)

    result_async_list = []
    def accumulate_result(result):
        # result_async_list.append(result)
        return

    def handle_error(exception):
        print("Exception ", exception)
        return

    for idx, data in enumerate(train_dataloader):

        fn = data['fn'][0]
        keypoints = data['keypoints']
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
                            (data, keypoints[:,person_id,:,:],process_queue),
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
        # print("Shape ", results[1]['result'].keys())
        # print("Shape ", results[1]['result']['camera_rotation'].shape,
        #     results[1]['result']['camera_translation'].shape,
        #     results[1]['result']['betas'].shape,
        #     results[1]['result']['body_pose'].shape,
        #     results[1]['result']['left_hand_pose'].shape)
        # print("List of result keys ", results)
        result_stored = {
            "idx":idx,
            "camera_translation":results[1]['result']['camera_translation'],
            "body_pose":results[1]['result']['body_pose'],
            "global_orient":results[1]['result']['global_orient'],
            "left_hand_pose":results[1]['result']['left_hand_pose'],
            "betas":results[1]['result']['betas'],
            "jaw_pose":results[1]['result']['jaw_pose'],
            "leye_pose":results[1]['result']['leye_pose'],
            "reye_pose":results[1]['result']['reye_pose'],
            "expression":results[1]['result']['expression'],
            "right_hand_pose":results[1]['result']['right_hand_pose']
        }
        if(result_stored['body_pose'][0,0] != result_stored['body_pose'][0,0]):
            print("Error in calculation ", idx, result_stored)
            # exit()
            result_stored = {
                "idx":idx,
                "camera_translation":data['cam_trans'],
                "body_pose":data['body_pose'][:,3:66],
                "global_orient":data['body_pose'][:,:3],
                "left_hand_pose":data['left_hand_pose'],
                "betas":data['mean_beta'],
                "right_hand_pose":data['right_hand_pose'],              
                "jaw_pose":np.zeros((batch_size, 3)),
                "leye_pose":np.zeros((batch_size, 3)),
                "reye_pose":np.zeros((batch_size, 3)),
                "expression":np.zeros((batch_size, 10)),
            }
        output_file = os.path.join(output_folder, str(idx)+".pkl")
        with open(output_file, 'wb') as handle:
            pickle.dump(result_stored, handle, protocol=pickle.HIGHEST_PROTOCOL)

        dataset_obj.update_new_idx(result_stored)

    args["batch_size"] = 1
    model_params['batch_size'] = 1


    return

def fit_single_frame(data, keypoints, process_queue, **kwargs):
    # print("Entering the process ", dir(process_queue))
    gpu_id = process_queue.get()
    # print("The current gpu id ", gpu_id)
    try:
        results = fit_single_frame_1(data, keypoints, gpu_id, **kwargs)
    finally:
        process_queue.put(gpu_id)
    return results
def fit_single_frame_1(data,
                     keypoints,
                     gpu_id,                     
                     model_params,
                     joint_weights,
                     gender=None,                     
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
                     rho=100,
                     use_joints_conf=False,
                     interactive=True,
                     visualize=False,
                     dtype=torch.float32,
                     **kwargs):
    # assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'
    device = torch.device('cuda:'+str(gpu_id)) if use_cuda else torch.device('cpu')
    joint_weights = joint_weights.to(device=device)
    batch_size = kwargs.get('batch_size')

    camera = cf.get_camera(kwargs.get('focal_length'), device, dtype, kwargs)

    body_model = cf.get_model(device, model_params, gender)

    body_pose_prior, jaw_prior, expr_prior, left_hand_prior, \
        right_hand_prior, shape_prior, angle_prior = \
            cf.get_priors(kwargs, device, dtype)

    opt_weights = cf.get_weights(data_weights, body_pose_prior_weights, 
                hand_pose_prior_weights, shape_weights, jaw_pose_prior_weights, 
                expr_weights, face_joints_weights, hand_joints_weights, 
                use_hands, use_face, device, dtype)

    # print("The input pose info ", data['mean_beta'].shape, 
    #     data['body_pose'].shape, body_model.betas.shape)
    betas = data['mean_beta'].float().to(device=device)
    body_pose = data['body_pose'].float().to(device=device)

    keypoint_data = keypoints.float().to(device=device)
    gt_joints = keypoint_data[:, :, :2]
    if use_joints_conf:
        joints_conf = keypoint_data[:, :, 2].reshape(batch_size, -1)

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

    monitor = pose_fitting.FittingMonitor(visualize=visualize, **kwargs)

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
    est_params['left_hand_pose'] = data['left_hand_pose'].float().to(device=device)
    est_params['right_hand_pose'] = data['right_hand_pose'].float().to(device=device)
    body_model.reset_params(**est_params)

    img = data['img'][0].float().to(device=device)

    H, W, _ = img.shape

    data_weight = 1000 / H
    
    # Update the value of the translation of the camera as well as
    # the image center.
    with torch.no_grad():
        camera.translation[:] = data['cam_trans'].view_as(camera.translation).float().to(device=device) #torch.tensor(data['cam_trans'], dtype=dtype) #init_t.view_as(camera.translation)
        camera.center[:] = torch.tensor([W, H], dtype=dtype).float().to(device=device) * 0.5
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
                    opt_idx, elapsed, final_loss_val/batch_size))


        # Get the result of the fitting process
        # Store in it the errors list in order to compare multiple
        # orientations, if they exist
        result = {'camera_' + str(key): val.detach().cpu().numpy()
                  for key, val in camera.named_parameters()}
        result.update({key: val.detach().cpu().numpy()
                       for key, val in body_model.named_parameters()})

        results.append({'loss': final_loss_val/batch_size,
                        'result': result})
        # print("Process ", results)

    return results
