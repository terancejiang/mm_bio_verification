#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Project Name: mm_bio_verification
File Created: 2024/3/5 下午3:01
Author: Ying.Jiang
File Name: Arcface.py
"""

"""
@author:Jun Wang
@date: 20201123
@contact: jun21wangustc@gmail.com
"""

import math
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter

from mmverification.registry import MODELS


@MODELS.register_module()
class ArcFace(Module):
    """Implementation for "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    """

    def __init__(self, feat_dim, num_class, margin_arc=0.35, margin_am=0.0, scale=32):
        super(ArcFace, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_class))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin_arc = margin_arc
        self.margin_am = margin_am
        self.scale = scale
        self.cos_margin = math.cos(margin_arc)
        self.sin_margin = math.sin(margin_arc)
        self.min_cos_theta = math.cos(math.pi - margin_arc)
        self.mm = math.sin(math.pi - margin_arc) * margin_arc  # dyj

    def forward(self, feats, labels):
        kernel_norm = F.normalize(self.weight, dim=0)

        feats = F.normalize(feats)

        # Matrix multiplication of features and weights, for cosine angle
        cos_theta = torch.mm(feats, kernel_norm)
        # remove numerical error, to ensure cos_theta is in range [-1, 1]
        cos_theta = cos_theta.clamp(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * self.cos_margin - sin_theta * self.sin_margin
        # 0 <= theta + m <= pi, ==> -m <= theta <= pi-m
        # because 0<=theta<=pi, so, we just have to keep theta <= pi-m, that is cos_theta >= cos(pi-m)
        ##########################################
        cos_theta_m = cos_theta_m.float()
        cos_theta = cos_theta.float()
        cos_theta_m = torch.where(cos_theta > self.min_cos_theta, cos_theta_m, cos_theta - self.mm)
        # cos_theta_m = torch.where(cos_theta > self.min_cos_theta, cos_theta_m, cos_theta-self.margin_am)
        ############################################
        index = torch.zeros_like(cos_theta)
        index.scatter_(1, labels.data.view(-1, 1), 1)
        index = index.byte().bool()
        output = cos_theta * 1.0
        output[index] = cos_theta_m[index]
        output *= self.scale
        return output
