"""""""""""""""""""""""""""""
Project: SPZ_insightface
Author: Terance Jiang
Date: 12/13/2023
"""""""""""""""""""""""""""""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
from typing import List
from torch import Tensor
import copy
import os

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False


class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class BasicStage(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 norm_layer,
                 act_layer,
                 pconv_fw_type
                 ):

        super().__init__()

        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x


class PatchEmbed(nn.Module):

    def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.proj(x))
        return x


class PatchMerging(nn.Module):

    def __init__(self, patch_size2, patch_stride2, dim, norm_layer, expend=True):
        super().__init__()
        if expend:
            outdim = 2 * dim
        else:
            outdim = dim
        self.reduction = nn.Conv2d(dim, outdim, kernel_size=patch_size2, stride=patch_stride2, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(outdim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.reduction(x))
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, groups=groups, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_c),
            # PReLU(num_parameters=out_c)
            nn.ReLU()
            # GELU()
        )

    def forward(self, x):
        return self.layers(x)
class GDC(nn.Module):
    def __init__(self, embedding_size):
        super(GDC, self).__init__()
        self.layers = nn.Sequential(
            LinearBlock(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0)),
            nn.Flatten(),
            nn.Linear(512, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size))

    def forward(self, x):
        return self.layers(x)
class LinearBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(LinearBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(num_features=out_c)
        )

    def forward(self, x):
        return self.layers(x)

class FasterNet(nn.Module):

    def __init__(self,
                 in_chans=1,
                 num_classes=1000,
                 embed_dim=96,
                 depths=(1, 2, 8, 2),
                 mlp_ratio=2.,
                 n_div=4,
                 patch_size=2,
                 patch_stride=2,
                 patch_size2=2,  # for subsequent layers
                 patch_stride2=2,
                 patch_norm=True,
                 feature_dim=1280,
                 drop_path_rate=0.1,
                 layer_scale_init_value=0,
                 norm_layer='BN',
                 act_layer='RELU',
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 pconv_fw_type='split_cat',
                 **kwargs):
        super().__init__()

        if norm_layer == 'BN':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError

        if act_layer == 'GELU':
            act_layer = nn.GELU
        elif act_layer == 'RELU':
            act_layer = partial(nn.ReLU, inplace=True)
        else:
            raise NotImplementedError

        width = [64,64,128,128]

        if not fork_feat:
            self.num_classes = num_classes
        self.num_stages = len(depths)
        self.embed_dim = width[0]
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))
        self.mlp_ratio = mlp_ratio
        self.depths = depths

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            patch_stride=patch_stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        # stochastic depth decay rule
        dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        stages_list = []
        i_stage_x = 0
        # for i_stage in range(self.num_stages):
        stage = BasicStage(dim=width[0],
                           n_div=n_div,
                           depth=1,
                           mlp_ratio=self.mlp_ratio,
                           drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],
                           layer_scale_init_value=layer_scale_init_value,
                           norm_layer=norm_layer,
                           act_layer=act_layer,
                           pconv_fw_type=pconv_fw_type
                           )
        stages_list.append(stage)

        # patch merging layer
        if 0 < self.num_stages - 1:
            stages_list.append(
                PatchMerging(patch_size2=patch_size2,
                             patch_stride2=patch_stride2,
                             dim=width[0],
                             norm_layer=norm_layer,
                             expend=False)
            )

        stage = BasicStage(dim=width[1],
                           n_div=n_div,
                           depth=2,
                           mlp_ratio=self.mlp_ratio,
                           drop_path=dpr[sum(depths[:1]):sum(depths[:1 + 1])],
                           layer_scale_init_value=layer_scale_init_value,
                           norm_layer=norm_layer,
                           act_layer=act_layer,
                           pconv_fw_type=pconv_fw_type
                           )
        stages_list.append(stage)

        # patch merging layer
        if 1 < self.num_stages - 1:
            stages_list.append(
                PatchMerging(patch_size2=patch_size2,
                             patch_stride2=patch_stride2,
                             dim=width[1],
                             norm_layer=norm_layer)
            )
        stage = BasicStage(dim=width[2],
                           n_div=n_div,
                           depth=8,
                           mlp_ratio=self.mlp_ratio,
                           drop_path=dpr[sum(depths[:2]):sum(depths[:2 + 1])],
                           layer_scale_init_value=layer_scale_init_value,
                           norm_layer=norm_layer,
                           act_layer=act_layer,
                           pconv_fw_type=pconv_fw_type
                           )
        stages_list.append(stage)

        # patch merging layer
        if 2 < self.num_stages - 1:
            stages_list.append(
                PatchMerging(patch_size2=patch_size2,
                             patch_stride2=patch_stride2,
                             dim=width[2],
                             norm_layer=norm_layer)
            )

        last_stage = BasicStage(dim=width[3],
                               n_div=n_div,
                               depth=1,
                               mlp_ratio=self.mlp_ratio,
                               drop_path=[dpr[-1]],
                               layer_scale_init_value=layer_scale_init_value,
                               norm_layer=norm_layer,
                               act_layer=act_layer,
                               pconv_fw_type=pconv_fw_type
                               )
        stages_list.append(last_stage)

        stages_list.append(
            PatchMerging(patch_size2=patch_size2,
                         patch_stride2=patch_stride2,
                         dim=width[3],
                         norm_layer=norm_layer,
                         expend=False)
        )

        self.stages = nn.Sequential(*stages_list)

        self.fork_feat = fork_feat

        if self.fork_feat:
            self.forward = self.forward_det
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    raise NotImplementedError
                else:
                    layer = norm_layer(int(embed_dim * 2 ** i_emb))
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            self.forward = self.forward_cls

            # Classifier head
            # self.avgpool_pre_head = nn.Sequential(
            #     nn.AdaptiveAvgPool2d(1),
            #     nn.Conv2d(self.num_features, feature_dim, 1, bias=False),
            #     act_layer()
            # )
            # self.head = nn.Linear(feature_dim, num_classes) \
            #     if num_classes > 0 else nn.Identity()
            self.conv_sep = ConvBlock(width[-1], 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
            self.features = GDC(512)

        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # init for mmdetection by loading imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)

            # show for debug
            print('missing_keys: ', missing_keys)
            print('unexpected_keys: ', unexpected_keys)

    def forward_cls(self, x):
        # output only the features of last layer for image classification
        x = self.patch_embed(x)
        x = self.stages(x)
        # x = self.avgpool_pre_head(x)  # B C 1 1
        x = self.conv_sep(x)
        x = self.features(x)


        return x

    def forward_det(self, x: Tensor) -> Tensor:
        # output the features of four stages for dense prediction
        x = self.patch_embed(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return outs

def fasternet_s(**kwargs):
    model = FasterNet(
        mlp_ratio=2,
        embed_dim=64,
        depths=(1, 2, 8),
        drop_path_rate=0.15,
        act_layer='RELU',
        fork_feat=False,
        **kwargs
        )

    return model

if __name__ == "__main__":
    from torch.optim import SGD
    from torch.cuda import amp
    import torch.nn as nn

    # from utils.torch_utils import select_device
    model = fasternet_s()

    from torchinfo import summary
    from torch.autograd import Variable

    summary(model, input_size=(1, 1, 112, 112))

    torch.save(model.state_dict(), '/zngjiangy/fasternet_s.pth')
