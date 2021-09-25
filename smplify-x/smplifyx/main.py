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

import time
import yaml
import torch

import smplx

from utils import JointMapper
from cmd_parser import parse_config
from data_parser import create_dataset
# from fit_single_frame import fit_single_frame
# from pose_refinement_parallel import refine_pose
from pose_refinement_batch_parallel import refine_pose

# from beta_refinement_1 import refine_beta
from beta_refinement_batch import refine_beta

from camera import create_camera
from prior import create_prior

torch.backends.cudnn.enabled = False

from video_keypoints_error import video_keypoints_error
import common_functions as cf



def main(**args):
    output_folder = args.pop('output_folder')
    output_folder = osp.expandvars(output_folder)
    if not osp.exists(output_folder):
        os.makedirs(output_folder)

    # Store the arguments for the current experiment
    conf_fn = osp.join(output_folder, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(args, conf_file)

    result_folder = args.pop('result_folder', 'results')
    result_folder = osp.join(output_folder, result_folder)
    if not osp.exists(result_folder):
        os.makedirs(result_folder)

    mesh_folder = args.pop('mesh_folder', 'meshes')
    mesh_folder = osp.join(output_folder, mesh_folder)
    if not osp.exists(mesh_folder):
        os.makedirs(mesh_folder)

    out_img_folder = osp.join(output_folder, 'images')
    if not osp.exists(out_img_folder):
        os.makedirs(out_img_folder)

    float_dtype = args['float_dtype']
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float64
    else:
        print('Unknown float type {}, exiting!'.format(float_dtype))
        sys.exit(-1)

    use_cuda = args.get('use_cuda', True)
    if use_cuda and not torch.cuda.is_available():
        print('CUDA is not available, exiting!')
        sys.exit(-1)

    img_folder = args.pop('img_folder', 'images')
    dataset_obj = create_dataset(img_folder=img_folder, **args)

    start = time.time()

    input_gender = args.pop('gender', 'neutral')
    gender_lbl_type = args.pop('gender_lbl_type', 'none')
    max_persons = args.pop('max_persons', -1)

    float_dtype = args.get('float_dtype', 'float32')
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Unknown float type {}, exiting!'.format(float_dtype))

    joint_mapper = JointMapper(dataset_obj.get_model2data())

    use_vposer = False #args.get('use_vposer', True)
    model_params = dict(model_path=args.get('model_folder'),
                        joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_body_pose=not use_vposer,
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        dtype=dtype,
                        # use_pca=False,
                        **args)

    # Create the camera object
    focal_length = args.get('focal_length')
    camera = create_camera(focal_length_x=focal_length,
                           focal_length_y=focal_length,
                           dtype=dtype,
                           **args)

    # print("CAMERA PARAMTERS ", camera.center, camera.focal_length_x,
    #     camera.focal_length_y)

    if hasattr(camera, 'rotation'):
        camera.rotation.requires_grad = False

    use_cuda =  args.get('use_cuda', True)
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        camera = camera.to(device=device)
    else:
        device = torch.device('cpu')
    # A weight for every joint of the model
    joint_weights = dataset_obj.get_joint_weights().to(device=device,
                                                       dtype=dtype)
    # Add a fake batch dimension for broadcasting
    joint_weights.unsqueeze_(dim=0)


    body_pose_prior, jaw_prior, expr_prior, left_hand_prior, \
        right_hand_prior, shape_prior, angle_prior = \
            cf.get_priors(args, device, dtype)

    # for idx, data in enumerate(train_dataloader):
    #     print("Input data ",  len(data['img']), len(data['fn']), idx,
    #         len(data['keypoints']), len(data['img_path']),
    #         data['img'][0].shape, data['keypoints'][0].shape)

    # exit()

    total_error = \
        video_keypoints_error(dataset_obj, result_folder, model_params,
         camera=camera,
         joint_weights=joint_weights,
         dtype=dtype,
         shape_prior=shape_prior,
         expr_prior=expr_prior,
         body_pose_prior=body_pose_prior,
         left_hand_prior=left_hand_prior,
         right_hand_prior=right_hand_prior,
         jaw_prior=jaw_prior,
         angle_prior=angle_prior,
         **args)

    for i in range(2):
        refine_pose(dataset_obj, result_folder, model_params,
             camera=camera,
             joint_weights=joint_weights,
             dtype=dtype,
             shape_prior=shape_prior,
             expr_prior=expr_prior,
             body_pose_prior=body_pose_prior,
             left_hand_prior=left_hand_prior,
             right_hand_prior=right_hand_prior,
             jaw_prior=jaw_prior,
             angle_prior=angle_prior,
             **args)

        total_error = \
            video_keypoints_error(dataset_obj, result_folder, model_params,
             camera=camera,
             joint_weights=joint_weights,
             dtype=dtype,
             shape_prior=shape_prior,
             expr_prior=expr_prior,
             body_pose_prior=body_pose_prior,
             left_hand_prior=left_hand_prior,
             right_hand_prior=right_hand_prior,
             jaw_prior=jaw_prior,
             angle_prior=angle_prior,
             **args)

        refine_beta(dataset_obj, result_folder, model_params,
             camera=camera,
             joint_weights=joint_weights,
             dtype=dtype,
             shape_prior=shape_prior,
             expr_prior=expr_prior,
             body_pose_prior=body_pose_prior,
             left_hand_prior=left_hand_prior,
             right_hand_prior=right_hand_prior,
             jaw_prior=jaw_prior,
             angle_prior=angle_prior,
             **args)

        total_error = \
            video_keypoints_error(dataset_obj, result_folder, model_params,
             camera=camera,
             joint_weights=joint_weights,
             dtype=dtype,
             shape_prior=shape_prior,
             expr_prior=expr_prior,
             body_pose_prior=body_pose_prior,
             left_hand_prior=left_hand_prior,
             right_hand_prior=right_hand_prior,
             jaw_prior=jaw_prior,
             angle_prior=angle_prior,
             **args)

    elapsed = time.time() - start
    time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                             time.gmtime(elapsed))
    print('Processing the data took: {}'.format(time_msg))


if __name__ == "__main__":
    args = parse_config()
    main(**args)
