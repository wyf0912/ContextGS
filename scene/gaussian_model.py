#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import math
import os
import time
from functools import reduce
from typing import Optional

import numpy as np
import torch
import compressai

from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn
from torch_scatter import scatter_max
from einops import repeat, rearrange

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.latent_codecs import LatentCodec, HyperLatentCodec

from utils.general_utils import (build_scaling_rotation, get_expon_lr_func,
                                 inverse_sigmoid, strip_symmetric)
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import mkdir_p
from utils.entropy_models import Entropy_bernoulli, Entropy_gaussian, Entropy_factorized
from utils.multi_level import torch_unique_with_indices
from utils.encodings import \
    STE_binary, STE_multistep, Quantize_anchor, \
    anchor_round_digits, \
    encoder, decoder, \
    encoder_gaussian, decoder_gaussian, \
    get_binary_vxl_size, Q_anchor


bit2MB_scale = 8 * 1024 * 1024

class GaussianModel(nn.Module):
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self,
                 feat_dim: int=32,
                 n_offsets: int=5,
                 voxel_size: float=0.01,
                 update_depth: int=3,
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank = False,
                 n_features_per_level: int=2,
                 resolutions_list=(18, 24, 33, 44, 59, 80, 108, 148, 201, 275, 376, 514),
                 resolutions_list_2D=(130, 258, 514, 1026),
                 ste_binary: bool=True,
                 ste_multistep: bool=False,
                 add_noise: bool=False,
                 Q=1,
                 use_2D: bool=True,
                 decoded_version: bool=False,
                 level_num: int=3,
                 adaptQ_per_channel: bool=False,
                 hyper_divisor: int=4,
                 target_ratio: float=None,
                 disable_hyper: bool=False
                 ):
        super().__init__()

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank
        self.x_bound_min = torch.zeros(size=[1, 3], device='cuda')
        self.x_bound_max = torch.ones(size=[1, 3], device='cuda')
        self.n_features_per_level = n_features_per_level

        self.resolutions_list = resolutions_list
        self.resolutions_list_2D = resolutions_list_2D
        self.ste_binary = ste_binary
        self.ste_multistep = ste_multistep
        self.add_noise = add_noise
        self.Q = Q
        self.use_2D = use_2D
        self.decoded_version = decoded_version
        self.level_num = level_num
        self.adaptQ_per_channel = adaptQ_per_channel
        self.hyper_divisor = hyper_divisor
        self.target_ratio = target_ratio
        self.level_scale = None
        self.disable_hyper = disable_hyper

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._mask = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        self._hyper_latent = torch.empty(0)

        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)

        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.latent_codec = EntropyBottleneck(channels=feat_dim//hyper_divisor).cuda()

        encoding_params_num = 0
        for n, p in self.latent_codec.named_parameters():
            encoding_params_num += p.numel()
        encoding_MB = encoding_params_num * 32 / 8 / 1024 / 1024
        print(f'Entropy_bottleneck={encoding_params_num}, size={encoding_MB}MB.')

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        mlp_input_feat_dim = feat_dim

        self.mlp_opacity = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()
        

        self.mlp_cov = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
            # nn.Linear(feat_dim, 7),
        ).cuda()
        

        self.mlp_color = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()
        

        self.mlp_grid = nn.ModuleList()
        # self.mlp_deform = nn.ModuleList()
        for i in range(level_num):
            feat_out_dim = (feat_dim+6+3*self.n_offsets)*2+1+1+1 # feat grid offset
            context_dim = self.feat_dim + 6 + 3
            feat_in_dim = self.feat_dim//hyper_divisor+3 if i==level_num-1 else context_dim+self.feat_dim//hyper_divisor

            self.mlp_grid.append(nn.Sequential(
                nn.Linear(feat_in_dim, feat_dim*2),
                nn.ReLU(True),
                nn.Linear(feat_dim*2, feat_out_dim),
            ).cuda())

        self.entropy_gaussian = Entropy_gaussian(Q=1).cuda()

    
    def get_mlp_size(self, digit=32):
        mlp_size = 0
        for n, p in self.named_parameters():
            if 'mlp' in n and 'deform' not in n:
                mlp_size += p.numel()*digit
        return mlp_size, mlp_size / 8 / 1024 / 1024

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        self.latent_codec.eval()
        self.mlp_grid.eval()
        # self.mlp_deform.eval()

        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        self.latent_codec.train()
        self.mlp_grid.train()
        # self.mlp_deform.train()

        if self.use_feat_bank:
            self.mlp_feature_bank.train()

    def capture(self):
        self.latent_codec.update()
        return (
            self._anchor,
            self._anchor_feat,
            self._hyper_latent,
            self._offset,
            self._mask,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            # self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            # mlp
            self.mlp_opacity.state_dict(),
            self.mlp_cov.state_dict(),
            self.mlp_color.state_dict(),
            self.latent_codec.state_dict(),
            self.mlp_grid.state_dict(),
            # bound
            self.x_bound_min,
            self.x_bound_max,
            # multi_scale
            self.level_scale
        )

    def restore(self, model_args, training_args):
        (
        self._anchor,
        self._anchor_feat,
        self._hyper_latent,
        self._offset,
        self._mask,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        # denom,
        opt_dict,
        self.spatial_lr_scale,
        # mlp
        mlp_opacity_state_dict,
        mlp_cov_state_dict,
        mlp_color_state_dict,
        latent_codec_state_dict,
        mlp_grid_state_dict,
        # bound
        self.x_bound_min,
        self.x_bound_max,
        # multi_scale
        self.level_scale
        ) = model_args
        self.latent_codec.update()
        self.training_setup(training_args)
        # self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        # load MLPs
        self.mlp_opacity.load_state_dict(mlp_opacity_state_dict)
        self.mlp_cov.load_state_dict(mlp_cov_state_dict)
        self.mlp_color.load_state_dict(mlp_color_state_dict)
        self.latent_codec.load_state_dict(latent_codec_state_dict)
        self.mlp_grid.load_state_dict(mlp_grid_state_dict)
        pass
        
    @property
    def get_scaling(self):
        if self.decoded_version:
            return self._scaling
        return 1.0*self.scaling_activation(self._scaling)

    @property
    def get_mask(self):
        if self.decoded_version:
            return self._mask
        mask_sig = torch.sigmoid(self._mask)
        return ((mask_sig > 0.01).float() - mask_sig).detach() + mask_sig

    @property
    def get_mask_anchor(self):
        with torch.no_grad():
            if self.decoded_version:
                mask_anchor = (torch.sum(self._mask, dim=1)[:, 0]) > 0
                return mask_anchor
            mask_sig = torch.sigmoid(self._mask)
            mask = ((mask_sig > 0.01).float() - mask_sig).detach() + mask_sig
            mask_anchor = (torch.sum(mask, dim=1)[:, 0]) > 0
            return mask_anchor

    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_grid_mlp(self):
        return self.mlp_grid

    @property
    def get_deform_mlp(self):
        return self.mlp_deform

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        if self.decoded_version:
            return self._anchor
        anchor, quantized_v = Quantize_anchor.apply(self._anchor, self.x_bound_min, self.x_bound_max)
        return anchor
    
    @property
    def min_Q_anchor(self):
        return ((self.x_bound_max - self.x_bound_min) / (2**15-1)).clamp(1e-5).detach()

    @torch.no_grad()
    def update_anchor_bound(self):
        x_bound_min = (torch.min(self._anchor, dim=0, keepdim=True)[0]).detach()
        x_bound_max = (torch.max(self._anchor, dim=0, keepdim=True)[0]).detach()
        for c in range(x_bound_min.shape[-1]):
            x_bound_min[0, c] = x_bound_min[0, c] * 1.2 if x_bound_min[0, c] < 0 else x_bound_min[0, c] * 0.8
        for c in range(x_bound_max.shape[-1]):
            x_bound_max[0, c] = x_bound_max[0, c] * 1.2 if x_bound_max[0, c] > 0 else x_bound_max[0, c] * 0.8
        self.x_bound_min = x_bound_min
        self.x_bound_max = x_bound_max
        print('anchor_bound_updated')

    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        return data

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        ratio = 1
        points = pcd.points[::ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')

        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        masks = torch.ones((fused_point_cloud.shape[0], self.n_offsets, 1)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        hyper_latent = torch.zeros((fused_point_cloud.shape[0], self.feat_dim//self.hyper_divisor)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._mask = nn.Parameter(masks.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._hyper_latent = nn.Parameter(hyper_latent.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._mask], 'lr': training_args.mask_lr_init * self.spatial_lr_scale, "name": "mask"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},

                {'params': self.mlp_grid.parameters(), 'lr': training_args.mlp_grid_lr_init, "name": "mlp_grid"},

                # {'params': self.mlp_deform.parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_deform"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._mask], 'lr': training_args.mask_lr_init * self.spatial_lr_scale, "name": "mask"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._hyper_latent], 'lr': training_args.hyper_latent_lr, "name": "hyper_latent"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},

                {'params': self.latent_codec.parameters(), 'lr': training_args.latent_codec_lr_init, "name": "latent_codec"},
                {'params': self.mlp_grid.parameters(), 'lr': training_args.mlp_grid_lr_init, "name": "mlp_grid"},

                # {'params': self.mlp_deform.parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_deform"},
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        self.mask_scheduler_args = get_expon_lr_func(lr_init=training_args.mask_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.mask_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.mask_lr_delay_mult,
                                                    max_steps=training_args.mask_lr_max_steps)

        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)

        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)

        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)

        self.latent_codec_scheduler_args = get_expon_lr_func(lr_init=training_args.latent_codec_lr_init,
                                                    lr_final=training_args.latent_codec_lr_final,
                                                    lr_delay_mult=training_args.latent_codec_lr_delay_mult,
                                                    max_steps=training_args.latent_codec_lr_max_steps,
                                                             step_sub=0 if self.ste_binary else 10000,
                                                             )
        self.mlp_grid_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_grid_lr_init,
                                                    lr_final=training_args.mlp_grid_lr_final,
                                                    lr_delay_mult=training_args.mlp_grid_lr_delay_mult,
                                                    max_steps=training_args.mlp_grid_lr_max_steps,
                                                         step_sub=0 if self.ste_binary else 10000,
                                                         )

        # self.mlp_deform_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_deform_lr_init,
        #                                             lr_final=training_args.mlp_deform_lr_final,
        #                                             lr_delay_mult=training_args.mlp_deform_lr_delay_mult,
        #                                             max_steps=training_args.mlp_deform_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mask":
                lr = self.mask_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "latent_codec":
                lr = self.latent_codec_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_grid":
                lr = self.mlp_grid_scheduler_args(iteration)
                param_group['lr'] = lr
            # if param_group["name"] == "mlp_deform":
            #     lr = self.mlp_deform_scheduler_args(iteration)
            #     param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._mask.shape[1]*self._mask.shape[2]):
            l.append('f_mask_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        for i in range(self._hyper_latent.shape[1]):
            l.append('f_hyper_latent_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        hyper_latent = self._hyper_latent.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        mask = self._mask.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, mask, anchor_feat, hyper_latent, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        # hyper_latent
        hyper_latent_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_hyper_latent")]
        hyper_latent_names = sorted(hyper_latent_names, key = lambda x: int(x.split('_')[-1]))
        hyper_latents = np.zeros((anchor.shape[0], len(hyper_latent_names)))
        for idx, attr_name in enumerate(hyper_latent_names):
            hyper_latents[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        mask_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_mask")]
        mask_names = sorted(mask_names, key = lambda x: int(x.split('_')[-1]))
        masks = np.zeros((anchor.shape[0], len(mask_names)))
        for idx, attr_name in enumerate(mask_names):
            masks[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        masks = masks.reshape((masks.shape[0], 1, -1))

        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self._hyper_latent = nn.Parameter(torch.tensor(hyper_latents, dtype=torch.float, device="cuda").requires_grad_(True))
        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._mask = nn.Parameter(torch.tensor(masks, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name'] or 'codec' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:  # Only for opacity, rotation. But seems they two are useless?
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        temp_opacity = temp_opacity.view([-1, self.n_offsets])

        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        self.anchor_demon[anchor_visible_mask] += 1

        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter

        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)

        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name'] or 'codec' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]


        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._mask = optimizable_tensors["mask"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._hyper_latent = optimizable_tensors["hyper_latent"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]


    def anchor_growing(self, grads, threshold, offset_mask):
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):  # 3
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)

            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)
            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:, :3].unsqueeze(dim=1)

            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor

            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)

            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)

                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1, 2]).float().cuda() * cur_size
                new_scaling = torch.log(new_scaling)

                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:, 0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_hyper_latent = self._hyper_latent.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim//self.hyper_divisor])[candidate_mask]
                new_hyper_latent = scatter_max(new_hyper_latent, inverse_indices.unsqueeze(1).expand(-1, new_hyper_latent.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()
                new_masks = torch.ones_like(candidate_anchor[:, 0:1]).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "hyper_latent": new_hyper_latent,
                    "offset": new_offsets,
                    "mask": new_masks,
                    "opacity": new_opacities,
                }

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()

                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._hyper_latent = optimizable_tensors["hyper_latent"]
                self._offset = optimizable_tensors["offset"]
                self._mask = optimizable_tensors["mask"]
                self._opacity = optimizable_tensors["opacity"]

    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)

        self.anchor_growing(grads_norm, grad_threshold, offset_mask)

        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask)  # [N]

        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        # update opacity accum
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()

        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self,path):
        self.latent_codec.update()
        mkdir_p(os.path.dirname(path))

        if self.use_feat_bank:
            torch.save({
                'opacity_mlp': self.mlp_opacity.state_dict(),
                'mlp_feature_bank': self.mlp_feature_bank.state_dict(),
                'cov_mlp': self.mlp_cov.state_dict(),
                'color_mlp': self.mlp_color.state_dict(),
                'latent_codec': self.latent_codec.state_dict(),
                'grid_mlp': self.mlp_grid.state_dict(),
                # 'deform_mlp': self.mlp_deform.state_dict(),
            }, path)
        else:
            torch.save({
                'opacity_mlp': self.mlp_opacity.state_dict(),
                'cov_mlp': self.mlp_cov.state_dict(),
                'color_mlp': self.mlp_color.state_dict(),
                'latent_codec': self.latent_codec.state_dict(),
                'grid_mlp': self.mlp_grid.state_dict(),
                'bound': [self.x_bound_min, self.x_bound_max],
                'level_scale': self.level_scale,
                # 'deform_mlp': self.mlp_deform.state_dict(),
            }, path)


    def load_mlp_checkpoints(self,path):
        checkpoint = torch.load(path)
        self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
        self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
        self.mlp_color.load_state_dict(checkpoint['color_mlp'])
        if self.use_feat_bank:
            self.mlp_feature_bank.load_state_dict(checkpoint['mlp_feature_bank'])
        self.latent_codec.update()
        self.latent_codec.load_state_dict(checkpoint['latent_codec'], strict=False)
        self.mlp_grid.load_state_dict(checkpoint['grid_mlp'])
        self.x_bound_min, self.x_bound_max = checkpoint['bound']
        self.level_scale = checkpoint['level_scale']
        # self.mlp_deform.load_state_dict(checkpoint['deform_mlp'])

    def contract_to_unisphere(self,
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: int = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            mask = mask.unsqueeze(-1) + 0.0
            x_c = (2 - 1 / mag) * (x / mag)
            x = x_c * mask + x * (1 - mask)
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x

    @torch.no_grad()
    def estimate_final_bits(self):
        mask_anchor = self.get_mask_anchor

        anchor = self.get_anchor[mask_anchor]
        feat = self._anchor_feat[mask_anchor]
        # TODO: bpp of hyper_latent
        grid_offsets = self._offset[mask_anchor]
        grid_scaling = self.get_scaling[mask_anchor]
        mask = self.get_mask[mask_anchor] # mask of gaussian points
        hyper = self._hyper_latent[mask_anchor]
        
        bit_anchor_sum, bit_hyper_sum, bit_feat_sum, bit_scaling_sum, bit_offsets_sum, bit_masks_sum = multi_scale_generating(self, anchor, hyper, feat, grid_offsets, grid_scaling, binary_grid_masks=mask, predict_bpp=True, return_sum_bits=True)
        
        log_info = f"\nEstimated sizes in MB: " \
                f"anchor {round(bit_anchor_sum/bit2MB_scale, 4)}, " \
                f"feat {round(bit_feat_sum/bit2MB_scale, 4)}, " \
                f"scaling {round(bit_scaling_sum/bit2MB_scale, 4)}, " \
                f"offsets {round(bit_offsets_sum/bit2MB_scale, 4)}, " \
                f"hyper {round(bit_hyper_sum/bit2MB_scale, 4)}, " \
                f"masks {round(bit_masks_sum/bit2MB_scale, 4)}, " \
                f"MLPs {round(self.get_mlp_size()[0]/bit2MB_scale, 4)}, " \
                f"Total {round((bit_anchor_sum + bit_feat_sum + bit_scaling_sum + bit_offsets_sum + bit_hyper_sum + bit_masks_sum + self.get_mlp_size()[0])/bit2MB_scale, 4)}"
        # print(log_info)
        return log_info


    @torch.no_grad()
    def conduct_encoding(self, pre_path_name):
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)
        torch.cuda.synchronize(); t1 = time.time()
        print('Start encoding ...')
        global t_codec; t_codec = 0
        content_pre_gathered = None
        to_code_hybrid_index_orign_space_list = []
        self.latent_codec.update(force=True)

        def encode_process(data, mean, scale, Q, file_name=None):
            global t_codec
            data = data.contiguous().view(-1)
            mean = mean.contiguous().view(-1)
            scale =  torch.clamp(scale.contiguous().view(-1), 1e-9)
            Q =Q.contiguous().view(-1)

            data = STE_multistep.apply(data, Q)
            torch.cuda.synchronize(); t0 = time.time()
            byte_stream, bit_data, min_data, max_data = encoder_gaussian(data, mean, scale, Q, file_name=file_name)
            torch.cuda.synchronize(); t_codec = t_codec + (time.time() - t0)
            return byte_stream, bit_data, min_data, max_data, data
        
        mask_anchor = self.get_mask_anchor
        # _anchor = self.get_anchor[mask_anchor]
        _anchor, quantized_anchor = Quantize_anchor.apply(self._anchor[mask_anchor], self.x_bound_min, self.x_bound_max)
        _feat = self._anchor_feat[mask_anchor]
        _grid_offsets = self._offset[mask_anchor]
        _scaling = self.get_scaling[mask_anchor]
        _mask = self.get_mask[mask_anchor]
        _hyper_latent = self._hyper_latent[mask_anchor]
        
        hyper_feat = self.latent_codec.quantize(_hyper_latent, mode="symbols", means=self.latent_codec._get_medians().permute(1,2,0)[0]) 
        # torch.use_deterministic_algorithms(False)
        if self.level_scale is None: self.level_scale = find_divide_scale(self, _anchor, self.target_ratio, self.level_num)
        hybrid_anchor_list, inverse_indices_list, mapping_list, hybrid_feat = divide_levels(self, _anchor, None)
        # torch.use_deterministic_algorithms(True)
        
        # context data
        feat_after_Q = torch.zeros_like(_feat)
        grid_scaling_after_Q = torch.zeros_like(_scaling)
        already_coded = torch.zeros_like(_feat[:,0]).bool()

        # metadata
        size_hyper_list = []
        hyper_bytes_list = []
        bit_anchor_list = []
        min_anchor_list = []
        max_anchor_list = []
        anchor_bytes_list = []
        
        bit_feat_list_dict = {}
        bit_scaling_list_dict = {}
        bit_offsets_list_dict = {}
        
        min_feat_list_dict = {}
        max_feat_list_dict = {}
        min_scaling_list_dict = {}
        max_scaling_list_dict = {}
        min_offsets_list_dict = {}
        max_offsets_list_dict = {}
        N_levels_list = []

        MAX_batch_size = 1_000

        # encode anchor
        anchor_b_name = os.path.join(pre_path_name, 'anchor.npy')
        hyper_b_name = os.path.join(pre_path_name, 'hyper.b')

        N_anchor = _anchor.shape[0]
        MAX_batch_size_for_anchor = MAX_batch_size * 10
        steps = (N_anchor // MAX_batch_size_for_anchor) if (N_anchor % MAX_batch_size_for_anchor) == 0 else (N_anchor // MAX_batch_size_for_anchor + 1)
        
        bit_hyper_list = []
        for s in range(steps):
            N_num = min(MAX_batch_size_for_anchor, N_anchor - s*MAX_batch_size_for_anchor)
            N_start = s * MAX_batch_size_for_anchor
            N_end = min((s+1)*MAX_batch_size_for_anchor, N_anchor)

            # encode hyper
            byte_stream_hyper = self.latent_codec.compress(rearrange(_hyper_latent[N_start:N_end], "b c->1 c b"))[0]
            bit_hyper = len(byte_stream_hyper)*8
            bit_hyper_list.append(bit_hyper)
            size_hyper_list.append(_hyper_latent[N_start:N_end].shape[0])
            hyper_bytes_list.append(byte_stream_hyper)
            
            torch.cuda.empty_cache()
        
        # save anchor
        bit_anchor_list.append(_anchor.numel() * 16)
        # interval = ((self.x_bound_max - self.x_bound_min) * Q_anchor + 1e-6)
        # quantized_anchor = torch.div((_anchor - self.x_bound_min), interval, rounding_mode='floor')
        np_anchor = quantized_anchor.cpu().numpy().astype(np.uint16)
        np.save(anchor_b_name, np_anchor)

        # # divide levels
        # if self.level_scale is None: self.level_scale = find_divide_scale(self, _anchor, self.target_ratio, self.level_num)
        # hybrid_anchor_list, inverse_indices_list, mapping_list, hybrid_feat = divide_levels(self, _anchor, None)

        with open(hyper_b_name, 'wb') as fout:
            bytes_stream = b"".join(hyper_bytes_list)
            fout.write(bytes_stream)

        # Encode multi-scale feature
        for level_idx in reversed(range(self.level_num)):
            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2

            min_feat_list = []
            max_feat_list = []
            min_scaling_list = []
            max_scaling_list = []
            min_offsets_list = []
            max_offsets_list = []
            
            bit_feat_list = []
            bit_scaling_list = []
            bit_offsets_list = []
            anchor_infos_list = []

            # encoded bytes
            feat_bytes_list = []
            scaling_bytes_list = []
            offsets_bytes_list = []

            # select the feature to be processed
            if level_idx == 0:
                hybrid_feat, hybrid_grid_scaling, hybrid_grid_offsets = _feat, _scaling, _grid_offsets
                mapping = torch.arange(_anchor.shape[0])
            else:
                mapping = mapping_to_orign(mapping_list, level_idx)
                hybrid_feat, hybrid_grid_scaling, hybrid_grid_offsets = _feat[mapping], _scaling[mapping], _grid_offsets[mapping]

            # filter already encoded ones
            if level_idx!=(self.level_num-1):
                already_enocoded_mapping = mapping_list[level_idx]
                to_code = torch.ones_like(hybrid_feat[:,0]).bool()
                to_code[already_enocoded_mapping] = False
            else:
                to_code = torch.ones_like(hybrid_feat[:,0]).bool()

            hybrid_feat = hybrid_feat[to_code]
            hybrid_grid_scaling = hybrid_grid_scaling[to_code]
            hybrid_grid_offsets = hybrid_grid_offsets[to_code]
            N_levels_list.append(hybrid_feat.shape[0])

            if level_idx != 0:
                to_code_hybrid_index_orign_space = mapping_to_orign(mapping_list, level_idx, to_code)
            else:
                to_code_hybrid_index_orign_space = torch.arange(_anchor.shape[0])[to_code]
            to_code_hybrid_index_orign_space_list.append(to_code_hybrid_index_orign_space) # for statics
            hybrid_anchor = hybrid_anchor_list[level_idx][to_code]

            # predict feature of the voxel
            if content_pre_gathered is None:
                feat_in = torch.cat([hybrid_anchor, hyper_feat[to_code_hybrid_index_orign_space].float()], dim=1)
            else:
                feat_in = torch.cat([content_pre_gathered, hyper_feat
                [to_code_hybrid_index_orign_space]], dim=1)
            predicted_attri = self.get_grid_mlp[level_idx](feat_in)

            # predict attribute based on the voxel feature
            mean_feat, scale_feat, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                        torch.split(predicted_attri, split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)  # [N_visible_anchor, 32], [N_visible_anchor, 32]


            Q_feat = (Q_feat * (1 + torch.tanh(Q_feat_adj))).repeat(1, mean_feat.shape[-1]).clamp(1e-9)
            Q_scaling = (Q_scaling * (1 + torch.tanh(Q_scaling_adj))).repeat(1, mean_scaling.shape[-1]).clamp(1e-9)
            Q_offsets = (Q_offsets * (1 + torch.tanh(Q_offsets_adj))).repeat(1, mean_offsets.shape[-1]).clamp(1e-9)           

            N = hybrid_anchor.shape[0]
            steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)
            
            feat_list = []
            scaling_list = []
            offsets_list = []

            masks_b_name = os.path.join(pre_path_name, 'masks.b')
            
            feat_b_name = os.path.join(pre_path_name, 'feat.b').replace('.b', f'{level_idx}.b')
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'{level_idx}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'{level_idx}.b')    

            for s in range(steps):
                N_num = min(MAX_batch_size, N - s*MAX_batch_size)
                N_start = s * MAX_batch_size
                N_end = min((s+1)*MAX_batch_size, N)

                anchor_infos = None
                anchor_infos_list.append(anchor_infos)

                # encode feature
                bytes_feat, bit_feat, min_feat, max_feat, feat = encode_process(hybrid_feat[N_start:N_end], mean_feat[N_start:N_end], scale_feat[N_start:N_end], Q_feat[N_start:N_end])
                bit_feat_list.append(bit_feat)
                min_feat_list.append(min_feat.cpu().int().item())
                max_feat_list.append(max_feat.cpu().int().item())
                feat_list.append(feat)
                feat_bytes_list.append(bytes_feat)

                # g=GaussianConditional(None).cuda()
                # indexes = g.build_indexes(scale_feat[N_start:N_end]*Q_feat[N_start:N_end])
                # assert (feat == decoder_gaussian(mean_feat[N_start:N_end].contiguous().view(-1), scale_feat[N_start:N_end].contiguous().view(-1).clamp(1e-9), Q_feat[N_start:N_end].contiguous().view(-1), bstream=bytes_feat, min_value=min_feat, max_value=max_feat)).all()

                # encode scaling
                bytes_scaling, bit_scaling, min_scaling, max_scaling, scaling = encode_process(hybrid_grid_scaling[N_start:N_end], mean_scaling[N_start:N_end], scale_scaling[N_start:N_end], Q_scaling[N_start:N_end])
                bit_scaling_list.append(bit_scaling)
                min_scaling_list.append(min_scaling.cpu().int().item())
                max_scaling_list.append(max_scaling.cpu().int().item())
                scaling_list.append(scaling)
                scaling_bytes_list.append(bytes_scaling)
                
                mask = _mask[to_code_hybrid_index_orign_space][N_start:N_end]  # {0, 1}  # [N_num, K, 1]
                mask = mask.repeat(1, 1, 3).view(-1, 3*self.n_offsets).view(-1).to(torch.bool)  # [N_num*K*3]

                # encode offsets
                offsets = hybrid_grid_offsets[N_start:N_end].view(-1, 3*self.n_offsets).view(-1)  # [N_num*K*3]
                bytes_offset, bit_offsets, min_offsets, max_offsets, offsets = encode_process(offsets[mask], mean_offsets[N_start:N_end].contiguous().view(-1)[mask], scale_offsets[N_start:N_end].contiguous().view(-1)[mask], Q_offsets[N_start:N_end].view(-1).contiguous()[mask])
                bit_offsets_list.append(bit_offsets)
                min_offsets_list.append(min_offsets.cpu().int().item())
                max_offsets_list.append(max_offsets.cpu().int().item())
                offsets_list.append(offsets)
                offsets_bytes_list.append(bytes_offset)

                torch.cuda.empty_cache()

            # Write to file
            for fb_name, bytes_list in [(hyper_b_name, hyper_bytes_list), (feat_b_name, feat_bytes_list), (scaling_b_name, scaling_bytes_list), (offsets_b_name, offsets_bytes_list)]:
                with open(fb_name, 'wb') as fout:
                    bytes_stream = b"".join(bytes_list)
                    fout.write(bytes_stream)

            feat_after_Q[to_code_hybrid_index_orign_space] = torch.cat(feat_list, dim=0).view(-1, self.feat_dim)
            grid_scaling_after_Q[to_code_hybrid_index_orign_space] = torch.cat(scaling_list, dim=0).view(-1, 6)
            already_coded[to_code_hybrid_index_orign_space] = True

            # Context feature
            if level_idx!=0:
                content_pre_gathered = extract_context_feat(_anchor, feat_after_Q, grid_scaling_after_Q, already_coded, inverse_indices_list, mapping_list, level_idx)
            
            bit_feat_list_dict[level_idx] = bit_feat_list
            bit_scaling_list_dict[level_idx] = bit_scaling_list
            bit_offsets_list_dict[level_idx] = bit_offsets_list

            min_feat_list_dict[level_idx] = min_feat_list
            max_feat_list_dict[level_idx] = max_feat_list
            min_scaling_list_dict[level_idx] = min_scaling_list
            max_scaling_list_dict[level_idx] = max_scaling_list
            min_offsets_list_dict[level_idx] = min_offsets_list
            max_offsets_list_dict[level_idx] = max_offsets_list

        bit_anchor = sum(bit_anchor_list)
        bit_hyper = sum(bit_hyper_list)
        bit_feat = sum([sum(bit_feat_list) for bit_feat_list in bit_feat_list_dict.values()])
        bit_scaling = sum([sum(bit_scaling_list) for bit_scaling_list in bit_scaling_list_dict.values()])
        bit_offsets = sum([sum(bit_offsets_list) for bit_offsets_list in bit_offsets_list_dict.values()])

        mask = _mask  # {0, 1}
        p = torch.zeros_like(mask).to(torch.float32)
        prob_masks = (mask.sum() / mask.numel()).item()
        p[...] = prob_masks
        bit_masks = encoder((mask * 2 - 1).view(-1), p.view(-1), file_name=masks_b_name)

        torch.cuda.synchronize(); t2 = time.time()
        print('encoding time:', t2 - t1)
        print('codec time:', t_codec)


        meta_path = os.path.join(pre_path_name, 'meta.b')
        torch.save([self._anchor.shape[0], MAX_batch_size, min_feat_list_dict, max_feat_list_dict, min_scaling_list_dict, max_scaling_list_dict, min_offsets_list_dict, max_offsets_list_dict, prob_masks, bit_hyper_list, bit_feat_list_dict, bit_scaling_list_dict, bit_offsets_list_dict, N_levels_list], meta_path)
        
        self.save_mlp_checkpoints(os.path.join(pre_path_name, 'mlp.pt'))
        with open(meta_path, "rb") as f:
            bit_meta = len(f.read()) * 8

        log_info = f"\nEncoded sizes in MB: " \
                f"meta {round(bit_meta/bit2MB_scale, 4)}, " \
                f"hyper {round(bit_hyper/bit2MB_scale, 4)}, " \
                f"anchor {round(bit_anchor/bit2MB_scale, 4)}, " \
                f"feat {round(bit_feat/bit2MB_scale, 4)}, " \
                f"scaling {round(bit_scaling/bit2MB_scale, 4)}, " \
                f"offsets {round(bit_offsets/bit2MB_scale, 4)}, " \
                f"masks {round(bit_masks/bit2MB_scale, 4)}, " \
                f"MLPs {round(self.get_mlp_size()[0]/bit2MB_scale, 4)}, " \
                f"Total {round((bit_meta + bit_hyper + bit_anchor + bit_feat + bit_scaling + bit_offsets + bit_masks + self.get_mlp_size()[0])/bit2MB_scale, 4)}, " \
                f"EncTime {round(t2 - t1, 4)}"
        
        return log_info
    

        
    @torch.no_grad()
    def conduct_decoding(self, pre_path_name):
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)
        torch.cuda.synchronize(); t1 = time.time()
        print('Start decoding ...')
        [N_full, MAX_batch_size, min_feat_list_dict, max_feat_list_dict, min_scaling_list_dict, max_scaling_list_dict, min_offsets_list_dict, max_offsets_list_dict, prob_masks, bit_hyper_list, bit_feat_list_dict, bit_scaling_list_dict, bit_offsets_list_dict, N_levels_list] = torch.load(os.path.join(pre_path_name, "meta.b"))
        self.load_mlp_checkpoints(os.path.join(pre_path_name, 'mlp.pt'))

        N_levels_list = list(reversed(N_levels_list))
        
        hyper_b_name = os.path.join(pre_path_name, 'hyper.b')
        anchor_b_name = os.path.join(pre_path_name, 'anchor.npy')

        with open(hyper_b_name, 'rb') as fin:
            byte_stream_hyper = fin.read()

        N_valid = sum(N_levels_list) 

        # decode hyper
        start_bit_hyper = 0
        MAX_batch_size_for_anchor = MAX_batch_size * 10
        hyper_decoded_list = []
        hyper_size_list = [MAX_batch_size_for_anchor for _ in range(N_valid//MAX_batch_size_for_anchor)]
        if N_valid//MAX_batch_size_for_anchor:
            hyper_size_list.append(N_valid%MAX_batch_size_for_anchor)
        steps = (N_valid // MAX_batch_size_for_anchor) if (N_valid % MAX_batch_size_for_anchor) == 0 else (N_valid // MAX_batch_size_for_anchor + 1)
        for s in range(steps):
            N_num = min(MAX_batch_size_for_anchor, N_valid - s*MAX_batch_size_for_anchor)
            N_start = s * MAX_batch_size_for_anchor
            N_end = min((s+1)*MAX_batch_size_for_anchor, N_valid)

            hyper_decoded = self.latent_codec.decompress([byte_stream_hyper[start_bit_hyper:start_bit_hyper+bit_hyper_list[s]//8]], [hyper_size_list[s]])
            hyper_decoded = rearrange(hyper_decoded, "1 c b -> b c")
            hyper_decoded_list.append(hyper_decoded)
            start_bit_hyper = start_bit_hyper+bit_hyper_list[s]//8
            
        hyper_decoded = torch.cat(hyper_decoded_list, dim=0)
        
        # decode anchor

        anchor_decoded = torch.tensor(np.load(anchor_b_name).astype(np.int32)).cuda()
        interval = ((self.x_bound_max - self.x_bound_min) * Q_anchor + 1e-6)
        anchor_decoded = anchor_decoded * interval + self.x_bound_min

        # torch.use_deterministic_algorithms(False)
        if self.level_scale is None: self.level_scale = find_divide_scale(self, anchor_decoded, self.target_ratio, self.level_num)
        hybrid_anchor_list, inverse_indices_list, mapping_list, _ = divide_levels(self, anchor_decoded, None)
        # torch.use_deterministic_algorithms(True)
        masks_b_name = os.path.join(pre_path_name, 'masks.b')
        p = torch.zeros(size=[N_valid, self.n_offsets, 1], device='cuda').to(torch.float32)
        p[...] = prob_masks
        masks_decoded = decoder(p.view(-1), masks_b_name)  # {-1, 1}
        masks_decoded = (masks_decoded + 1) / 2  # {0, 1}
        masks_decoded = masks_decoded.view(-1, self.n_offsets, 1)   
        
        # for debug
        to_code_hybrid_index_orign_space_list = []
        
        # context data
        content_pre_gathered = None
        already_coded = torch.zeros_like(anchor_decoded[:,0]).float().bool()
        feat_after_Q = torch.zeros_like(anchor_decoded[:,[0]].float().repeat(1, self.feat_dim))
        grid_scaling_after_Q = torch.zeros_like(anchor_decoded[:,[0]].float().repeat(1, 6))
        grid_offset_after_Q = torch.zeros_like(anchor_decoded[:,[0]].float().repeat(1, 3*self.n_offsets)).view(-1, self.n_offsets, 3)
        debug_list = []


        for i in reversed(range(self.level_num)):
            # filter already encoded ones
            if i!=(self.level_num-1):
                already_enocoded_mapping = mapping_list[i]
                to_code = torch.ones_like(hybrid_anchor_list[i][:,0]).bool()
                to_code[already_enocoded_mapping] = False
            else:
                to_code = torch.ones_like(hybrid_anchor_list[i][:,0]).bool()
            if i != 0:
                to_code_hybrid_index_orign_space = mapping_to_orign(mapping_list, i, to_code)
            else:
                to_code_hybrid_index_orign_space = torch.arange(anchor_decoded.shape[0])[to_code]
            to_code_hybrid_index_orign_space_list.append(to_code_hybrid_index_orign_space)
            
            feat_decoded_list = []
            scaling_decoded_list = []
            offsets_decoded_list = []

            feat_b_name = os.path.join(pre_path_name, 'feat.b').replace('.b', f'{i}.b')
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'{i}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'{i}.b')

            with open(feat_b_name, 'rb') as fin:
                byte_stream_feat = fin.read()
            with open(scaling_b_name, 'rb') as fin:
                byte_stream_scaling = fin.read()
            with open(offsets_b_name, 'rb') as fin:
                byte_stream_offsets = fin.read()

            bit_feat_list = bit_feat_list_dict[i]
            bit_scaling_list = bit_scaling_list_dict[i]
            bit_offsets_list = bit_offsets_list_dict[i]

            min_feat_list = min_feat_list_dict[i]
            max_feat_list = max_feat_list_dict[i]
            min_scaling_list = min_scaling_list_dict[i]
            max_scaling_list = max_scaling_list_dict[i]
            min_offsets_list = min_offsets_list_dict[i]
            max_offsets_list = max_offsets_list_dict[i]

            start_bit_hyper = start_bit_feat = start_bit_scaling = start_bit_offsets = 0
            N_i = N_levels_list[i]
            steps = (N_i // MAX_batch_size) if (N_i % MAX_batch_size) == 0 else (N_i // MAX_batch_size + 1)

            
            if content_pre_gathered is None:
                feat_in = torch.cat([anchor_decoded[to_code_hybrid_index_orign_space], hyper_decoded[to_code_hybrid_index_orign_space].float()], dim=1)
            else:
                feat_in = torch.cat([content_pre_gathered, hyper_decoded[to_code_hybrid_index_orign_space]], dim=1)
            predicted_attri = self.get_grid_mlp[i](feat_in)

            # predict attribute based on the voxel feature
            mean_feat, scale_feat, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                        torch.split(predicted_attri, split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)  # [N_visible_anchor, 32], [N_visible_anchor, 32]
            
            for s in range(steps):
                min_feat = min_feat_list[s]
                max_feat = max_feat_list[s]
                min_scaling = min_scaling_list[s]
                max_scaling = max_scaling_list[s]
                min_offsets = min_offsets_list[s]
                max_offsets = max_offsets_list[s]
                
                N_num = min(MAX_batch_size, N_i - s*MAX_batch_size)
                N_start = s * MAX_batch_size
                N_end = min((s+1)*MAX_batch_size, N_i)
                # sizes of MLPs is not included here

                Q_feat = 1
                Q_scaling = 0.001
                Q_offsets = 0.2

                Q_feat_adj_ = Q_feat_adj[N_start:N_end].contiguous().repeat(1, mean_feat.shape[-1]).view(-1)
                Q_scaling_adj_ = Q_scaling_adj[N_start:N_end].contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
                Q_offsets_adj_ = Q_offsets_adj[N_start:N_end].contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)

                mean_feat_ = mean_feat[N_start:N_end].contiguous().view(-1)
                mean_scaling_ = mean_scaling[N_start:N_end].contiguous().view(-1)
                mean_offsets_ = mean_offsets[N_start:N_end].contiguous().view(-1)
                
                scale_feat_ = torch.clamp(scale_feat[N_start:N_end].contiguous().view(-1), min=1e-9)
                scale_scaling_ = torch.clamp(scale_scaling[N_start:N_end].contiguous().view(-1), min=1e-9)
                scale_offsets_ = torch.clamp(scale_offsets[N_start:N_end].contiguous().view(-1), min=1e-9)
                
                Q_feat = (Q_feat * (1 + torch.tanh(Q_feat_adj_))).clamp(1e-9)
                Q_scaling = (Q_scaling * (1 + torch.tanh(Q_scaling_adj_))).clamp(1e-9)
                Q_offsets = (Q_offsets * (1 + torch.tanh(Q_offsets_adj_))).clamp(1e-9)

                # decode feat
                feat_decoded = decoder_gaussian(mean_feat_, scale_feat_, Q_feat, bstream=byte_stream_feat[start_bit_feat: start_bit_feat+bit_feat_list[s]//8], min_value=min_feat, max_value=max_feat)
                feat_decoded = feat_decoded.view(N_num, self.feat_dim)  # [N_num, 32]
                start_bit_feat += bit_feat_list[s]//8
                feat_decoded_list.append(feat_decoded)

                # decode scaling
                scaling_decoded = decoder_gaussian(mean_scaling_, scale_scaling_, Q_scaling, bstream=byte_stream_scaling[start_bit_scaling: start_bit_scaling+bit_scaling_list[s]//8], min_value=min_scaling, max_value=max_scaling)
                scaling_decoded = scaling_decoded.view(N_num, 6)  # [N_num, 6]
                scaling_decoded_list.append(scaling_decoded)
                start_bit_scaling += bit_scaling_list[s]//8

                # decode offset
                masks_tmp = masks_decoded[to_code_hybrid_index_orign_space][N_start:N_end].repeat(1, 1, 3).view(-1, 3 * self.n_offsets).view(-1).to(torch.bool)

                offsets_decoded_tmp = decoder_gaussian(mean_offsets_[masks_tmp], scale_offsets_[masks_tmp], Q_offsets[masks_tmp], bstream=byte_stream_offsets[start_bit_offsets: start_bit_offsets+bit_offsets_list[s]//8], min_value=min_offsets, max_value=max_offsets)
                offsets_decoded = torch.zeros_like(mean_offsets_)
                offsets_decoded[masks_tmp] = offsets_decoded_tmp
                offsets_decoded = offsets_decoded.view(N_num, -1).view(N_num, self.n_offsets, 3)  # [N_num, K, 3]
                offsets_decoded_list.append(offsets_decoded)
                start_bit_offsets += bit_offsets_list[s]//8

                torch.cuda.empty_cache()

            assert start_bit_feat == len(byte_stream_feat)
            assert start_bit_scaling == len(byte_stream_scaling)
            assert start_bit_offsets == len(byte_stream_offsets)

            
            feat_decoded = torch.cat(feat_decoded_list, dim=0)
            scaling_decoded = torch.cat(scaling_decoded_list, dim=0)
            offsets_decoded = torch.cat(offsets_decoded_list, dim=0)

            # idx_in_orign = mapping_to_orign(mapping_list, i) if i >=1 else torch.arange(_anchor.shape[0])
            # to_code_mask = torch.zeros_like(_anchor[:,0]).bool()
            # to_code_mask[idx_in_orign] = True
            # to_code_mask = to_code_mask & (~already_coded)
            feat_after_Q[to_code_hybrid_index_orign_space] = feat_decoded
            grid_scaling_after_Q[to_code_hybrid_index_orign_space] = scaling_decoded
            grid_offset_after_Q[to_code_hybrid_index_orign_space] = offsets_decoded
            # Context feature
            if i!=0:
                already_coded[to_code_hybrid_index_orign_space] = True
                content_pre_gathered = extract_context_feat(anchor_decoded, feat_after_Q, grid_scaling_after_Q, already_coded, inverse_indices_list, mapping_list, i)
                
            torch.cuda.synchronize(); t2 = time.time()
            debug_list.append(hyper_decoded)
        print('decoding time:', t2 - t1)
        # fill back N_full
        _hyper = torch.zeros(size=[N_full, self.feat_dim//self.hyper_divisor], device='cuda')
        _anchor = torch.zeros(size=[N_full, 3], device='cuda')
        _anchor_feat = torch.zeros(size=[N_full, self.feat_dim], device='cuda')
        _offset = torch.zeros(size=[N_full, self.n_offsets, 3], device='cuda')
        _scaling = torch.zeros(size=[N_full, 6], device='cuda')
        _mask = torch.zeros(size=[N_full, self.n_offsets, 1], device='cuda')
        
        _anchor[:N_valid] = anchor_decoded
        _hyper[:N_valid] = hyper_decoded
        _anchor_feat[:N_valid] = feat_after_Q
        _offset[:N_valid] = grid_offset_after_Q
        _scaling[:N_valid] = grid_scaling_after_Q
        _mask[:N_valid] = masks_decoded

        print('Start replacing parameters with decoded ones...')
        # replace attributes by decoded ones
        # assert self._hyper_latent.shape == _hyper.shape
        self._hyper_latent = nn.Parameter(_hyper)
        # assert self._anchor_feat.shape == _anchor_feat.shape
        self._anchor_feat = nn.Parameter(_anchor_feat)
        # assert self._offset.shape == _offset.shape
        self._offset = nn.Parameter(_offset)
        # If change the following attributes, decoded_version must be set True
        self.decoded_version = True
        # assert self.get_anchor.shape == _anchor.shape
        self._anchor = nn.Parameter(_anchor)
        # assert self._scaling.shape == _scaling.shape
        self._scaling = nn.Parameter(_scaling)
        # assert self._mask.shape == _mask.shape
        self._mask = nn.Parameter(_mask)

        print('Parameters are successfully replaced by decoded ones!')

        log_info = f"\nDecTime {round(t2 - t1, 4)}"

        return log_info

def multi_scale_generating(pc, anchor, hyper, feat, grid_offsets, grid_scaling, binary_grid_masks, mask_anchor_bool=None, training=False, predict_bpp=False, return_sum_bits=False):
    content_pre_gathered = None
    to_code_hybrid_index_orign_space_list = []

    data_for_vis = {}

    feat_after_Q = torch.zeros_like(feat)
    grid_scaling_after_Q = torch.zeros_like(grid_scaling)
    grid_offsets_after_Q = torch.zeros_like(grid_offsets)
    already_coded = torch.zeros_like(feat[:,0]).bool()
    if predict_bpp:
        mean_feat_all, scale_feat_all, Q_feat_all, mean_scaling_all, scale_scaling_all, Q_scaling_all, mean_offsets_all, scale_offsets_all, Q_offsets_all  = torch.zeros_like(feat), torch.zeros_like(feat), torch.zeros_like(feat[:,[0]]), torch.zeros_like(grid_scaling), torch.zeros_like(grid_scaling), torch.zeros_like(grid_scaling[:,[0]]), torch.zeros_like(grid_offsets.view(-1, 3*pc.n_offsets)), torch.zeros_like(grid_offsets.view(-1, 3*pc.n_offsets)), torch.zeros_like(grid_offsets.view(-1, 3*pc.n_offsets)[:,[0]])
    
    assert isinstance(pc.latent_codec, EntropyBottleneck)

    hyper_feat, likelihood_hyper = pc.latent_codec(hyper, training=training)
    if pc.disable_hyper: hyper_feat = hyper_feat * 0
    
    if pc.level_scale is None: pc.level_scale = find_divide_scale(pc, anchor[mask_anchor_bool], pc.target_ratio, pc.level_num)
    hybrid_anchor_list, inverse_indices_list, mapping_list, hybrid_anchor = divide_levels(pc, anchor, mask_anchor_bool)
    
    for i in reversed(range(pc.level_num)):
        # Q_anchor = 0.01
        Q_feat = 1
        Q_scaling = 0.001
        Q_offsets = 0.2

        # select the feature to be processed
        if i == 0:
            hybrid_feat, hybrid_grid_scaling, hybrid_grid_offsets = feat, grid_scaling, grid_offsets
            mapping = torch.arange(anchor.shape[0])
        else:
            mapping = mapping_to_orign(mapping_list, i)
            hybrid_feat, hybrid_grid_scaling, hybrid_grid_offsets = feat[mapping], grid_scaling[mapping], grid_offsets[mapping]

        # filter already encoded ones
        if i!=(pc.level_num-1):
            already_enocoded_mapping = mapping_list[i]
            to_code = torch.ones_like(hybrid_feat[:,0]).bool()
            to_code[already_enocoded_mapping] = False
        else:
            to_code = torch.ones_like(hybrid_anchor[:,0]).bool()
        if to_code.sum() > 0:
            hybrid_feat = hybrid_feat[to_code]
            hybrid_grid_scaling = hybrid_grid_scaling[to_code]
            hybrid_grid_offsets = hybrid_grid_offsets[to_code]
            if i != 0:
                to_code_hybrid_index_orign_space = mapping_to_orign(mapping_list, i, to_code)
            else:
                to_code_hybrid_index_orign_space = torch.arange(anchor.shape[0])[to_code]
            to_code_hybrid_index_orign_space_list.append(to_code_hybrid_index_orign_space) # for statics

            hybrid_anchor = hybrid_anchor_list[i][to_code]
                
            # predict feature of the voxel
            if content_pre_gathered is None:
                feat_in = torch.cat([hybrid_anchor, hyper_feat[to_code_hybrid_index_orign_space].float()], dim=1)
            else:
                feat_in = torch.cat([content_pre_gathered, hyper_feat[to_code_hybrid_index_orign_space]], dim=1)
            predicted_attri = pc.get_grid_mlp[i](feat_in)

            # predict attribute based on the voxel feature
            mean_feat, scale_feat, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                        torch.split(predicted_attri, split_size_or_sections=[pc.feat_dim, pc.feat_dim, 6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1, 1], dim=-1)  # [N_visible_anchor, 32], [N_visible_anchor, 32]
            
            Q_feat = (Q_feat * (1 + torch.tanh(Q_feat_adj))).clamp(1e-9)
            Q_scaling = (Q_scaling * (1 + torch.tanh(Q_scaling_adj))).clamp(1e-9)
            Q_offsets = (Q_offsets * (1 + torch.tanh(Q_offsets_adj))).clamp(1e-9)

            if training:
                hybrid_feat = hybrid_feat + torch.empty_like(hybrid_feat).uniform_(-0.5, 0.5) * Q_feat
                hybrid_grid_scaling = hybrid_grid_scaling + torch.empty_like(hybrid_grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
                if pc.adaptQ_per_channel:
                    hybrid_grid_offsets = hybrid_grid_offsets + torch.empty_like(hybrid_grid_offsets).uniform_(-0.5, 0.5) * Q_offsets.view(hybrid_feat.shape[0], pc.n_offsets, -1) # .unsqueeze(1)
                else:
                    hybrid_grid_offsets = hybrid_grid_offsets + torch.empty_like(hybrid_grid_offsets).uniform_(-0.5, 0.5) * Q_offsets.unsqueeze(1)
            else:
                hybrid_feat = (STE_multistep.apply(hybrid_feat, Q_feat)).detach()
                hybrid_grid_scaling = (STE_multistep.apply(hybrid_grid_scaling, Q_scaling)).detach()
                if pc.adaptQ_per_channel:
                    hybrid_grid_offsets = (STE_multistep.apply
                    (hybrid_grid_offsets, Q_offsets.view(hybrid_feat.shape[0], pc.n_offsets, -1))).detach()
                else:
                    hybrid_grid_offsets = (STE_multistep.apply
                    (hybrid_grid_offsets, Q_offsets.unsqueeze(1))).detach()
            hybrid_grid_offsets = hybrid_grid_offsets.view(-1, 3*pc.n_offsets)

            # predict bpp
            if predict_bpp:
                mean_feat_all[to_code_hybrid_index_orign_space] = mean_feat
                scale_feat_all[to_code_hybrid_index_orign_space] = scale_feat
                Q_feat_all[to_code_hybrid_index_orign_space] = Q_feat

                mean_scaling_all[to_code_hybrid_index_orign_space] = mean_scaling
                scale_scaling_all[to_code_hybrid_index_orign_space] = scale_scaling
                Q_scaling_all[to_code_hybrid_index_orign_space] = Q_scaling

                mean_offsets_all[to_code_hybrid_index_orign_space] = mean_offsets
                scale_offsets_all[to_code_hybrid_index_orign_space] = scale_offsets
                Q_offsets_all[to_code_hybrid_index_orign_space] = Q_offsets


            # context feature
            feat_after_Q[to_code_hybrid_index_orign_space] = hybrid_feat
            grid_scaling_after_Q[to_code_hybrid_index_orign_space] = hybrid_grid_scaling
            grid_offsets_after_Q[to_code_hybrid_index_orign_space] = hybrid_grid_offsets.view(-1, pc.n_offsets, 3)
            already_coded[to_code_hybrid_index_orign_space] = True
        
        # Context feature
        if i!=0:
            content_pre_gathered = extract_context_feat(anchor, feat_after_Q, grid_scaling_after_Q, already_coded, inverse_indices_list, mapping_list, i)
            data_for_vis.update({f"g{i}": content_pre_gathered})

    if not predict_bpp:
        return feat_after_Q, grid_scaling_after_Q, grid_offsets_after_Q
    else:
        # calculate bit map
        chosse_random_thresh = 1 if return_sum_bits else 0.15
        choose_mask = torch.rand_like(anchor[:, 0]) <= chosse_random_thresh
        if mask_anchor_bool is not None:
            choose_mask = choose_mask & mask_anchor_bool
            mask_anchor_rate = (mask_anchor_bool.sum() / mask_anchor_bool.numel()).detach()
        else:
            mask_anchor_rate = 1

        bit_hyper = -torch.log2(likelihood_hyper[choose_mask])
        bit_feat = pc.entropy_gaussian.forward(feat_after_Q[choose_mask], mean_feat_all[choose_mask], scale_feat_all[choose_mask], Q_feat_all[choose_mask], pc._anchor_feat.mean())
        bit_scaling = pc.entropy_gaussian.forward(grid_scaling_after_Q[choose_mask], mean_scaling_all[choose_mask], scale_scaling_all[choose_mask], Q_scaling_all[choose_mask], pc.get_scaling.mean())
        bit_offsets = pc.entropy_gaussian.forward(grid_offsets_after_Q[choose_mask].view(-1, 3*pc.n_offsets), mean_offsets_all[choose_mask], scale_offsets_all[choose_mask], Q_offsets_all[choose_mask], pc._offset.mean())
        bit_offsets = bit_offsets * binary_grid_masks[choose_mask].repeat(1, 1, 3).view(-1, 3*pc.n_offsets)

        if return_sum_bits:
            data_for_vis.update({
                "a0" : anchor[to_code_hybrid_index_orign_space_list[0]],
                "a1" : anchor[to_code_hybrid_index_orign_space_list[1]],
                "a2" : anchor[to_code_hybrid_index_orign_space_list[2]],
                "f2" : feat_after_Q[to_code_hybrid_index_orign_space_list[2]],
                "f1": feat_after_Q[to_code_hybrid_index_orign_space_list[1]],
                "sum": bit_feat.sum(dim=1)
            })
            torch.save(data_for_vis, "data_for_vis.pt")
            torch.save(bit_feat.sum(dim=1), "bit.pt")
            bit_anchor = bit_hyper.shape[0] * 3 * 16
            bit_masks_sum = get_binary_vxl_size(binary_grid_masks)[1].item()        
            return bit_anchor, torch.sum(bit_hyper).item(), torch.sum(bit_feat).item(), torch.sum(bit_scaling).item(), torch.sum(bit_offsets).item(), bit_masks_sum
    
        # average bpp
        bit_per_anchor_param = 16 * mask_anchor_rate
        bit_per_hyper_param = torch.sum(bit_hyper) / bit_hyper.numel() * mask_anchor_rate
        bit_per_feat_param =  torch.sum(bit_feat) / bit_feat.numel() * mask_anchor_rate
        bit_per_scaling_param = torch.sum(bit_scaling) / bit_scaling.numel() * mask_anchor_rate
        bit_per_offsets_param = torch.sum(bit_offsets) / bit_offsets.numel() * mask_anchor_rate
        bit_per_param = (torch.sum(bit_feat) + torch.sum(bit_scaling) + torch.sum(bit_offsets) + torch.sum(bit_hyper)) / \
                        (bit_feat.numel() + bit_scaling.numel() + bit_offsets.numel()) * mask_anchor_rate 
        
        # predict bpp of each level
        bpp_sum_map = (bit_offsets.sum(dim=1) + bit_scaling.sum(dim=1) + bit_feat.sum(dim=1))
        feat_dim = pc.feat_dim + 6 + 3*pc.n_offsets
        each_level_bpp = [1-mask_anchor_bool.float().mean().item(), bit_per_hyper_param.item()]
        for index in to_code_hybrid_index_orign_space_list:
            level_mask = torch.zeros_like(anchor[:,0]).bool()
            level_mask[index] = True 
            ratio = index.shape[0]/anchor.shape[0]
            bpp = bpp_sum_map[level_mask[choose_mask]].mean().item() / feat_dim
            each_level_bpp.append([ratio, bpp])

        return feat_after_Q, grid_scaling_after_Q, grid_offsets_after_Q, bit_per_param, bit_per_feat_param, bit_per_scaling_param, bit_per_offsets_param, each_level_bpp
    

    
def extract_context_feat(anchor_after_Q, feat_after_Q, grid_scaling_after_Q, already_coded, inverse_indices_list, mapping_list, i):
    content_coded = torch.cat([anchor_after_Q, feat_after_Q, grid_scaling_after_Q], dim=1)
            # content_pre = content_coded[mapping_to_orign[mapping_list, i]]
    if i > 1:
        to_be_gathered_mask = torch.zeros_like(content_coded[:,0]).bool()
        to_be_gathered_index = mapping_to_orign(mapping_list,i-1)
        to_be_gathered_mask[to_be_gathered_index] = True
    else:
        to_be_gathered_mask = torch.ones_like(content_coded[:,0]).bool()
    to_be_gathered_mask = to_be_gathered_mask & (~already_coded)
    last_level_index_in_orign = index_of_level_L_in_orign(mapping_list, inverse_indices_list, torch.nonzero(to_be_gathered_mask)[:,0], i)
    content_pre_gathered = torch.gather(content_coded, dim=0, 
                                            index=repeat(last_level_index_in_orign, f"n->n {content_coded.shape[1]}"))
    return content_pre_gathered

def find_divide_scale(pc, anchor, target_ratio, level_num):
    scale_upper = ((pc.x_bound_max - pc.x_bound_min) / pc.voxel_size).max()
    
    def binary_search(scale_upper, scale_lower, anchor, target_ratio):
        while True:
            scale = (scale_upper+scale_lower)/2
            anchor_unique = torch.unique(torch.round(anchor/pc.voxel_size/scale), dim=0)*pc.voxel_size*scale
            ratio = anchor_unique.shape[0]/anchor.shape[0]
            if abs(ratio-target_ratio)<0.01 or (scale_upper - scale_lower).abs()<1:
                break
            if ratio < target_ratio:
                scale_upper = scale
            else:
                scale_lower = scale
        return scale, anchor_unique
    
    anchor_unique = anchor
    scale_list = []
    scale_lower = 1
    for i in range(level_num-1):
        scale, anchor_unique = binary_search(scale_upper, scale_lower, anchor_unique, target_ratio)
        scale_lower = scale
        scale_list.append(scale.item())
    return scale_list

def divide_levels(pc, anchor, mask_anchor_bool=None):
    hybrid_anchor_list = [anchor]
    inverse_indices_list = []
    mapping_list = []        
    hybrid_anchor = anchor
    for i in range(1, pc.level_num):
        # hyper-prior feature (level i)
        if i ==1 and mask_anchor_bool is not None:
            hybrid_anchor = hybrid_anchor*mask_anchor_bool.unsqueeze(1)
        unique_anchor, inverse_indices, mapping, counts = torch_unique_with_indices(torch.round(hybrid_anchor/pc.voxel_size/pc.level_scale[i-1]),dim=0) 
        hybrid_anchor = hybrid_anchor[mapping]
        hybrid_anchor_list.append(hybrid_anchor)
        inverse_indices_list.append(inverse_indices)
        mapping_list.append(mapping)
    return hybrid_anchor_list,inverse_indices_list,mapping_list,hybrid_anchor
    

def mapping_to_orign(mapping_list, L, mask=None):
    """ Mapping the index of level $L$ to the original space. 

    Args:
        mapping_list (list of tensor): the list of mapping from level $i+1$ to level $i$
        L (int): the starting level
        mask (binary mask tensor, optional): The mask of used indexes in level $L$. Defaults to use all indexes in level $L$.

    Returns:
        The index in the orginal space (the space w/o multi-scale)
    """
    assert L>0, "If L=0, the orgin space can be directly obtained"
    level = L - 1
    if mask is None:
        mapping_prev = mapping_list[level]
    else:
        mapping_prev = mapping_list[level][mask]
    for i in reversed(range(level)):
        mapping_prev = mapping_list[i][mapping_prev]
    return mapping_prev

def index_of_level_L_in_orign(mapping_list, inverse_indices_list, to_be_gathered_index, L):
    tmp = to_be_gathered_index
    for i in range(L):
        tmp = inverse_indices_list[i][tmp]
    return mapping_to_orign(mapping_list, L, mask=tmp)