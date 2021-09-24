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

import time

import numpy as np

import torch
import torch.nn as nn

from mesh_viewer import MeshViewer
import utils
from torch.utils.data import DataLoader




class FittingMonitor(object):
    def __init__(self, summary_steps=1, visualize=False,
                 maxiters=100, ftol=2e-09, gtol=1e-05,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='smpl',
                 **kwargs):
        super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.visualize = visualize
        self.summary_steps = summary_steps
        self.body_color = body_color
        self.model_type = model_type

    def __enter__(self):
        self.steps = 0
        if self.visualize:
            self.mv = MeshViewer(body_color=self.body_color)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.visualize:
            self.mv.close_viewer()

    def set_colors(self, vertex_color):
        batch_size = self.colors.shape[0]

        self.colors = np.tile(
            np.array(vertex_color).reshape(1, 3),
            [batch_size, 1])

    def run_fitting(self, optimizer, closure, params, body_model,
                    **kwargs):
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized
                body_model: nn.Module
                    The body model PyTorch module
                use_vposer: bool
                    Flag on whether to use VPoser (default=True).
                pose_embedding: torch.tensor, BxN
                    The tensor that contains the latent pose variable.
                vposer: nn.Module
                    The VPoser module
            Returns
            -------
                loss: float
                The final loss value
        '''
        append_wrists = self.model_type == 'smpl' and use_vposer
        prev_loss = None
        # print("List of params ", params)
        for n in range(self.maxiters):
            loss = optimizer.step(closure)
            print("Total Loss ", loss, self.maxiters, n)

            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = utils.rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    break

            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                # print(([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                #     for var in params if var.grad is not None]))
                # print("Breaking now ", self.gtol, params[0].grad,
                #     params[0])
                break

            prev_loss = loss.item()

            print("Loss ", n, loss)

        return prev_loss

    def create_fitting_closure(self,
                               optimizer, body_model, camera=None,
                               loss=None,
                               joint_weights=None,
                               return_verts=True, return_full_pose=False,
                               create_graph=False,
                               dataset_obj=None,
                               dtype=None,
                               use_joints_conf=True,
                               **kwargs):
        faces_tensor = body_model.faces_tensor.view(-1)
        train_dataloader = DataLoader(dataset_obj, batch_size=1, shuffle=False)
        def fitting_func(backward=True):
            torch.autograd.set_detect_anomaly(True)
            # print("Append wrist ", optimizer, body_model)
            device = joint_weights.device

            person_id = 0
            total_loss = 0.0
            for idx, data in enumerate(train_dataloader):
                print("Betas ", idx, body_model.betas)

                # print("The data keys ", data.keys(), data['body_pose'].shape)
                # body_pose = torch.unsqueeze(data['body_pose'], 0).float().to(device=device)
                body_pose = data['body_pose'].float().to(device=device)

                est_params = {}
                # est_params['betas'] = betas
                est_params['body_pose'] = body_pose[:, 3:66]
                est_params['global_orient'] = body_pose[:, :3]

                # Update the value of the translation of the camera as well as
                # the image center.
                with torch.no_grad():
                    camera.translation[:] = data['cam_trans'].view_as(camera.translation) #torch.tensor(data['cam_trans'], dtype=dtype) #init_t.view_as(camera.translation)

                body_model.reset_params(**est_params)
                print("After reset ", idx, body_model.betas)

                keypoints = data['keypoints'][0]
                # print('Processing: {}'.format(data['img_path']))
                keypoints = keypoints[[person_id]]
                keypoint_data = torch.tensor(keypoints, dtype=dtype)
                gt_joints = keypoint_data[:, :, :2]
                gt_joints = gt_joints.to(device=device, dtype=dtype)
                if use_joints_conf:
                    joints_conf = keypoint_data[:, :, 2].reshape(1, -1)
                    joints_conf = joints_conf.to(device=device, dtype=dtype)

                if backward:
                    optimizer.zero_grad()

                body_model_output = body_model(return_verts=return_verts,
                                               return_full_pose=return_full_pose)
                model_loss = loss(body_model_output, camera=camera,
                                  gt_joints=gt_joints,
                                  joints_conf=joints_conf,
                                  joint_weights=joint_weights,
                                  **kwargs)
                if backward:
                    model_loss.backward(create_graph=create_graph)
                print("Model loss ", model_loss, body_model_output.betas)
                total_loss += model_loss

            self.steps += 1
            if self.visualize and self.steps % self.summary_steps == 0:
                model_output = body_model_1(return_verts=True,
                                          body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            return torch.tensor(total_loss)

        return fitting_func


def create_loss(loss_type='smplify', **kwargs):
    if loss_type == 'smplify':
        return SMPLifyLoss(**kwargs)
    elif loss_type == 'camera_init':
        return SMPLifyCameraInitLoss(**kwargs)
    else:
        raise ValueError('Unknown loss type: {}'.format(loss_type))


class SMPLifyLoss(nn.Module):

    def __init__(self, 
                 rho=100,
                 body_pose_prior=None,
                 shape_prior=None,
                 use_joints_conf=True,
                 dtype=torch.float32,
                 data_weight=1.0,
                 body_pose_weight=0.0,
                 shape_weight=0.0):

        super(SMPLifyLoss, self).__init__()

        self.use_joints_conf = use_joints_conf

        self.robustifier = utils.GMoF(rho=rho)
        self.rho = rho

        self.body_pose_prior = body_pose_prior

        self.shape_prior = shape_prior

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer('body_pose_weight',
                             torch.tensor(body_pose_weight, dtype=dtype))
        self.register_buffer('shape_weight',
                             torch.tensor(shape_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                    weight_tensor = loss_weight_dict[key].clone().detach()
                else:
                    weight_tensor = torch.tensor(loss_weight_dict[key],
                                                 dtype=weight_tensor.dtype,
                                                 device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints, 
        joints_conf, joint_weights, **kwargs):
        projected_joints = camera(body_model_output.joints)
        # print("Shapes of output ", projected_joints.shape,
        #     body_model_output.joints.shape,
        #     joint_weights.shape, joints_conf.shape,
        #     gt_joints.shape, body_model_output.body_pose.shape)
        # Calculate the weights for each joints
        weights = (joint_weights * joints_conf
                   if self.use_joints_conf else
                   joint_weights).unsqueeze(dim=-1)

        # Calculate the distance of the projected joints from
        # the ground truth 2D detections
        joint_diff = self.robustifier(gt_joints - projected_joints)
        joint_loss = (torch.sum(weights ** 2 * joint_diff) *
                      self.data_weight ** 2)

        pprior_loss = torch.sum(self.body_pose_prior(
                body_model_output.body_pose,
                body_model_output.betas)) * self.body_pose_weight ** 2

        shape_loss = torch.sum(self.shape_prior(
            body_model_output.betas)) * self.shape_weight ** 2

        total_loss = (joint_loss + pprior_loss + shape_loss)
        # print("Loss jl ", joint_loss, " ppl ", pprior_loss, " sl ", shape_loss)
        # print("Weights ppw ", self.data_weight, self.body_pose_weight)
        # print("Total loss ", total_loss)
        return total_loss

