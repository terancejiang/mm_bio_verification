#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Project Name: mm_bio_verification
File Created: 2024/3/5 下午3:10
Author: Ying.Jiang
File Name: basic_face_model.py
"""
import torch
from mmengine.model import BaseModel
from mmverification.models.utils import initialize_weights
from mmverification.registry import MODELS


@MODELS.register_module()
class FaceModel(BaseModel):
    """Define a traditional face model which contains a backbone and a head.

    Attributes:
        backbone(object): the backbone of face model.
        head(object): the head of face model.
    """

    def __init__(self, backbone,
                 head,
                 init_weights=False,
                 with_head=True):
        """Init face model by backbone factory and head factory.

        Args:
            backbone(object): produce a backbone according to config files.
            head(object): produce a head according to config files.
        """
        super(FaceModel, self).__init__()
        self.backbone = backbone
        self.head = head
        self.with_head = with_head

        if init_weights:
            initialize_weights(self.backbone)

    def forward(self, data=None, label=None, is_train=True, **kwargs):
        if is_train and self.with_head:
            feat = self.backbone.forward(data, **kwargs)
            pred = self.head.forward(feat, label)
            return feat, pred
        else:
            feat = self.backbone.forward(data)
            return feat
