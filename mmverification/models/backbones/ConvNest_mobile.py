"""""""""""""""""""""""""""""
Project: Face_Recognition_full_pipeline
Author: Terance Jiang
Date: 10/23/2023
"""""""""""""""""""""""""""""
# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, type='gemm'):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


class Block(nn.Module):
    r""" ConvNeSt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        res_scale_init_value (float): Init value for Residual Scale. Default: 1.0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., res_scale_init_value=1.0, layer_scale_init_value=1e-6, type='gemm',er=12):
        super().__init__()
        self.pwconv1 = nn.Conv2d(dim, er * dim, kernel_size=1, stride=1,
                                 padding=0)  # pointwise/1x1 convs, implemented with Conv2d layers
        self.act1 = nn.GELU()
        self.norm1 = LayerNorm(er * dim, eps=1e-6, data_format="channels_first")
        self.dwconv = get_conv2d(er * dim, er * dim, kernel_size=5, stride=1, padding=2, dilation=1, groups=er * dim,
                                 bias=True, type=type)  # depthwise conv
        self.act2 = nn.GELU()
        self.pwconv2 = nn.Conv2d(er * dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm2 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.res_scale = nn.Parameter(res_scale_init_value * torch.ones((dim, 1, 1)),
                                      requires_grad=True) if res_scale_init_value else None
        # self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)),
        #                             requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        if self.res_scale is not None:
            input = self.res_scale * input

        x = self.pwconv1(x)
        x = self.act1(x)

        inter = x
        x = self.norm1(x)
        x = self.dwconv(x) + inter

        x = self.act2(x)
        x = self.pwconv2(x)
        x = self.norm2(x)

        x = input + self.drop_path(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvNeSt(nn.Module):
    r""" ConvNeSt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        res_scale_init_values (list): Init value for Residual Scale. Default: [None, None, 1.0, 1.0].
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=1, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 res_scale_init_values=[None, None, 1.0, 1.0], layer_scale_init_value=1e-6, head_init_scale=1.,
                 type='gemm',er=12
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=2, stride=2, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], er=er,
                        res_scale_init_value=res_scale_init_values[i], layer_scale_init_value=layer_scale_init_value,
                        type=type) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer

        self.head = nn.Linear(dims[-1], 512)

        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

        # self.head = nn.Sequential(
        #     nn.Conv2d(dims[-1], 512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        #     LayerNorm(dims[-1], eps=1e-6, data_format="channels_first"),
        #     nn.GELU(),
        #
        #     nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        #     Flatten(),
        #     nn.Linear(512, 512, bias=False),
        #     LayerNorm(512, eps=1e-6, data_format="channels_first"),
        # )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # if isinstance(m, (nn.Conv2d, nn.Linear)):
        #     trunc_normal_(m.weight, std=.02)
        #     nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnest_tiny_1k": "https://dl.fbaipublicfiles.com/convnest/convnest_tiny_1k_224_ema.pth",
    "convnest_small_1k": "https://dl.fbaipublicfiles.com/convnest/convnest_small_1k_224_ema.pth",
    "convnest_base_1k": "https://dl.fbaipublicfiles.com/convnest/convnest_base_1k_224_ema.pth",
    "convnest_large_1k": "https://dl.fbaipublicfiles.com/convnest/convnest_large_1k_224_ema.pth",
    "convnest_tiny_22k": "https://dl.fbaipublicfiles.com/convnest/convnest_tiny_22k_224.pth",
    "convnest_small_22k": "https://dl.fbaipublicfiles.com/convnest/convnest_small_22k_224.pth",
    "convnest_base_22k": "https://dl.fbaipublicfiles.com/convnest/convnest_base_22k_224.pth",
    "convnest_large_22k": "https://dl.fbaipublicfiles.com/convnest/convnest_large_22k_224.pth",
    "convnest_xlarge_22k": "https://dl.fbaipublicfiles.com/convnest/convnest_xlarge_22k_224.pth",
}


@register_model
def convnest_attom(pretrained=False, in_22k=False, **kwargs):
    # model = ConvNeSt(depths=[1, 1, 6, 2], dims=[32, 32, 64, 128],er=8, **kwargs)
    model = ConvNeSt(depths=[2, 2, 6, 2], dims=[32, 64, 128, 128],er=8, **kwargs)

    if pretrained:
        url = model_urls['convnest_atto_22k'] if in_22k else model_urls['convnest_atto_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnest_femto(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeSt(depths=[2, 2, 9, 2], dims=[24, 48, 96, 192], **kwargs)
    if pretrained:
        url = model_urls['convnest_femto_22k'] if in_22k else model_urls['convnest_femto_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnest_pico(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeSt(depths=[2, 2, 9, 2], dims=[32, 64, 128, 256], **kwargs)
    if pretrained:
        url = model_urls['convnest_pico_22k'] if in_22k else model_urls['convnest_pico_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnest_nano(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeSt(depths=[2, 2, 12, 2], dims=[40, 80, 160, 320], **kwargs)
    if pretrained:
        url = model_urls['convnest_nano_22k'] if in_22k else model_urls['convnest_nano_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnest_tiny(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeSt(depths=[3, 3, 14, 3], dims=[48, 96, 192, 384], **kwargs)
    if pretrained:
        url = model_urls['convnest_tiny_22k'] if in_22k else model_urls['convnest_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnest_small(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeSt(depths=[3, 3, 24, 3], dims=[56, 112, 224, 448], **kwargs)
    if pretrained:
        url = model_urls['convnest_small_22k'] if in_22k else model_urls['convnest_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnest_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeSt(depths=[3, 3, 28, 3], dims=[72, 144, 288, 576], **kwargs)
    if pretrained:
        url = model_urls['convnest_base_22k'] if in_22k else model_urls['convnest_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnest_large2(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeSt(depths=[3, 3, 26, 3], dims=[112, 224, 448, 896], **kwargs)
    if pretrained:
        url = model_urls['convnest_large_22k'] if in_22k else model_urls['convnest_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


if __name__ == '__main__':
    import torchsummary

    net = convnest_attom()
    # # You need to define input size to calcualte parameters
    # torchsummary.summary(net, input_size=(3, 112, 112))

    import torchinfo

    # net = MobileNetV3_Small_Face()
    # You need to define input size to calcualte parameters
    # torchinfo.summary(net, input_size=(3, 112, 112),batch_dim=0)
    block = Block(20)
    torchinfo.summary(net, input_size=(1, 112, 112), batch_dim=0)
