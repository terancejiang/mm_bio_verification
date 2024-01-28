# Copyright (c) OpenMMLab. All rights reserved.
import random
from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseDataPreprocessor, ImgDataPreprocessor
from mmengine.structures import PixelData
from torch import Tensor

from mmverification.registry import MODELS
from mmverification.utils import ConfigType

try:
    import skimage
except ImportError:
    skimage = None


@MODELS.register_module()
class BioDataPreprocessor(ImgDataPreprocessor):
    """Data preprocessor for image classification.

    Args:
        bgr_to_rgb (bool): Whether to convert the image from BGR to RGB,
            used in image transform. Default: True.
        color_type (str): The color type of input image. Options are `color`,
            `grayscale` and `unchanged`. Default: 'color'.
        mean (Sequence[float]): Mean values of 3 channels.
            Default: (0.485, 0.456, 0.406).
        std (Sequence[float]): Std values of 3 channels.
            Default: (0.229, 0.224, 0.225).
        rgb_to_bgr (bool): Whether to convert the image from RGB to BGR,
            used in image transform. Default: False.
        """
    def __init__(self,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 non_blocking: Optional[bool] = False,
                 ):
        super().__init__(
            mean=mean,
            std=std,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            non_blocking=non_blocking
        )

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization,padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`BioDataSample`): The class and annotation info of the sample.
        """

        inputs, data_samples = super().forward(data, training)

        return inputs, data_samples



@MODELS.register_module()
class MultiBranchDataPreprocessor(BaseDataPreprocessor):
    """DataPreprocessor wrapper for multi-branch data.

    Take semi-supervised object detection as an example, assume that
    the ratio of labeled data and unlabeled data in a batch is 1:2,
    `sup` indicates the branch where the labeled data is augmented,
    `unsup_teacher` and `unsup_student` indicate the branches where
    the unlabeled data is augmented by different pipeline.

    The input format of multi-branch data is shown as below :

    .. code-block:: none
        {
            'inputs':
                {
                    'sup': [Tensor, None, None],
                    'unsup_teacher': [None, Tensor, Tensor],
                    'unsup_student': [None, Tensor, Tensor],
                },
            'data_sample':
                {
                    'sup': [DetDataSample, None, None],
                    'unsup_teacher': [None, DetDataSample, DetDataSample],
                    'unsup_student': [NOne, DetDataSample, DetDataSample],
                }
        }

    The format of multi-branch data
    after filtering None is shown as below :

    .. code-block:: none
        {
            'inputs':
                {
                    'sup': [Tensor],
                    'unsup_teacher': [Tensor, Tensor],
                    'unsup_student': [Tensor, Tensor],
                },
            'data_sample':
                {
                    'sup': [DetDataSample],
                    'unsup_teacher': [DetDataSample, DetDataSample],
                    'unsup_student': [DetDataSample, DetDataSample],
                }
        }

    In order to reuse `DetDataPreprocessor` for the data
    from different branches, the format of multi-branch data
    grouped by branch is as below :

    .. code-block:: none
        {
            'sup':
                {
                    'inputs': [Tensor]
                    'data_sample': [DetDataSample, DetDataSample]
                },
            'unsup_teacher':
                {
                    'inputs': [Tensor, Tensor]
                    'data_sample': [DetDataSample, DetDataSample]
                },
            'unsup_student':
                {
                    'inputs': [Tensor, Tensor]
                    'data_sample': [DetDataSample, DetDataSample]
                },
        }

    After preprocessing data from different branches,
    the multi-branch data needs to be reformatted as:

    .. code-block:: none
        {
            'inputs':
                {
                    'sup': [Tensor],
                    'unsup_teacher': [Tensor, Tensor],
                    'unsup_student': [Tensor, Tensor],
                },
            'data_sample':
                {
                    'sup': [DetDataSample],
                    'unsup_teacher': [DetDataSample, DetDataSample],
                    'unsup_student': [DetDataSample, DetDataSample],
                }
        }

    Args:
        data_preprocessor (:obj:`ConfigDict` or dict): Config of
            :class:`DetDataPreprocessor` to process the input data.
    """

    def __init__(self, data_preprocessor: ConfigType) -> None:
        super().__init__()
        self.data_preprocessor = MODELS.build(data_preprocessor)

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization,padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor`` for multi-branch data.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict:

            - 'inputs' (Dict[str, obj:`torch.Tensor`]): The forward data of
                models from different branches.
            - 'data_sample' (Dict[str, obj:`DetDataSample`]): The annotation
                info of the sample from different branches.
        """

        if training is False:
            return self.data_preprocessor(data, training)

        # Filter out branches with a value of None
        for key in data.keys():
            for branch in data[key].keys():
                data[key][branch] = list(
                    filter(lambda x: x is not None, data[key][branch]))

        # Group data by branch
        multi_branch_data = {}
        for key in data.keys():
            for branch in data[key].keys():
                if multi_branch_data.get(branch, None) is None:
                    multi_branch_data[branch] = {key: data[key][branch]}
                elif multi_branch_data[branch].get(key, None) is None:
                    multi_branch_data[branch][key] = data[key][branch]
                else:
                    multi_branch_data[branch][key].append(data[key][branch])

        # Preprocess data from different branches
        for branch, _data in multi_branch_data.items():
            multi_branch_data[branch] = self.data_preprocessor(_data, training)

        # Format data by inputs and data_samples
        format_data = {}
        for branch in multi_branch_data.keys():
            for key in multi_branch_data[branch].keys():
                if format_data.get(key, None) is None:
                    format_data[key] = {branch: multi_branch_data[branch][key]}
                elif format_data[key].get(branch, None) is None:
                    format_data[key][branch] = multi_branch_data[branch][key]
                else:
                    format_data[key][branch].append(
                        multi_branch_data[branch][key])

        return format_data

    @property
    def device(self):
        return self.data_preprocessor.device

    def to(self, device: Optional[Union[int, torch.device]], *args,
           **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Args:
            device (int or torch.device, optional): The desired device of the
                parameters and buffers in this module.

        Returns:
            nn.Module: The model itself.
        """

        return self.data_preprocessor.to(device, *args, **kwargs)

    def cuda(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """

        return self.data_preprocessor.cuda(*args, **kwargs)

    def cpu(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """

        return self.data_preprocessor.cpu(*args, **kwargs)


