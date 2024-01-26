'''
Adapted from https://github.com/cavalleria/cavaface.pytorch/blob/master/backbone/mobilefacenet.py
Original author cavalleria
'''

import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module, ReLU, GELU
import torch
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F


class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvBlock(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            Conv2d(in_c, out_c, kernel, groups=groups, stride=stride, padding=padding, bias=False),
            # BatchNorm2d(num_features=out_c),
            LayerNorm(out_c, eps=1e-6),
            # PReLU(n
            # um_parameters=out_c)
            ReLU()
            # GELU()
        )

    def forward(self, x):
        return self.layers(x)


class LinearBlock(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(LinearBlock, self).__init__()
        self.layers = nn.Sequential(
            Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
            BatchNorm2d(num_features=out_c)
        )

    def forward(self, x):
        return self.layers(x)


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


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, out_dim, drop_path=0., layer_scale_init_value=1e-6, dim_scale=3, stride=(1, 1), groups=1):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, stride=stride)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim_scale * dim)  # pointwise/1x1 convs, implemented with linear layers
        # self.act = nn.GELU()
        self.act = nn.ReLU()
        self.pwconv2 = nn.Linear(dim_scale * dim, out_dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        if input.shape == x.shape:
            x = input + self.drop_path(x)
        return x


class DepthWise(Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(DepthWise, self).__init__()
        self.residual = residual
        self.layers = nn.Sequential(
            ConvBlock(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1)),
            ConvBlock(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride),
            LinearBlock(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        )

    def forward(self, x):
        short_cut = None
        if self.residual:
            short_cut = x
        x = self.layers(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Block(dim=c,out_dim=c, drop_path=0, layer_scale_init_value=1e-6, groups=c))
        self.layers = Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class GDC(Module):
    def __init__(self, embedding_size):
        super(GDC, self).__init__()
        self.layers = nn.Sequential(
            LinearBlock(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0)),
            nn.Flatten(),
            Linear(512, embedding_size, bias=False),
            BatchNorm1d(embedding_size))

    def forward(self, x):
        return self.layers(x)
class ConvBlock(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            Conv2d(in_c, out_c, kernel, groups=groups, stride=stride, padding=padding, bias=False),
            BatchNorm2d(num_features=out_c),
            # PReLU(num_parameters=out_c)
            ReLU()
            # GELU()
        )

    def forward(self, x):
        return self.layers(x)

class MobileFaceNet(Module):
    def __init__(self, fp16=False, num_features=512, blocks=(1, 4, 6, 2), scale=1):
        super(MobileFaceNet, self).__init__()
        self.scale = scale
        self.fp16 = fp16
        self.layers = nn.ModuleList()

        stem0 = nn.Sequential(
            nn.Conv2d(1, 64 * self.scale, kernel_size=3, stride=2, padding=1),
            LayerNorm(64 * self.scale, eps=1e-6, data_format="channels_first"),
            nn.ReLU(),
        )
        stem1 = nn.Sequential(
            # nn.Conv2d(64 * self.scale, 64 * self.scale, kernel_size=3, stride=2, padding=1),
            Block(dim=64 * self.scale, out_dim=64 * self.scale, drop_path=0, layer_scale_init_value=1e-6, stride=(2, 2), groups=128),
            # LayerNorm(64 * self.scale, eps=1e-6, data_format="channels_first")
        )
        stem2 = nn.Sequential(
            Block(dim=64 * self.scale, out_dim=128 * self.scale, drop_path=0, layer_scale_init_value=1e-6, stride=(2, 2), groups=256),

            # nn.Conv2d(64 * self.scale, 128 * self.scale, kernel_size=3, stride=2, padding=1),
            # LayerNorm(128 * self.scale, eps=1e-6, data_format="channels_first")
        )
        stem3 = nn.Sequential(
            Block(dim=128 * self.scale, out_dim=128 * self.scale, drop_path=0, layer_scale_init_value=1e-6,
                  stride=(2, 2), groups=512),

            # nn.Conv2d(128 * self.scale, 128 * self.scale, kernel_size=3, stride=2, padding=1),
            # LayerNorm(128 * self.scale, eps=1e-6, data_format="channels_first")
        )

        self.layers.append(
            # ConvBlock(1, 64 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
            stem0
        )

        # if blocks[0] == 1:
        self.layers.append(
            # ConvBlock(64 * self.scale, 64 * self.scale, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
            # Block(dim=64 * self.scale,out_dim=64 * self.scale, drop_path=0, layer_scale_init_value=1e-6, groups=64)
            ConvBlock(64 * self.scale, 64 * self.scale, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)

        )

        self.layers.extend(
            [
                # DepthWise(64 * self.scale, 64 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128),
                stem1,
                Residual(64 * self.scale, num_block=blocks[1], groups=128, kernel=(3, 3), stride=(1, 1),
                         padding=(1, 1)),
                # DepthWise(64 * self.scale, 128 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256),
                stem2,
                Residual(128 * self.scale, num_block=blocks[2], groups=256, kernel=(3, 3), stride=(1, 1),
                         padding=(1, 1)),
                # DepthWise(128 * self.scale, 128 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512),
                stem3,
                Residual(128 * self.scale, num_block=blocks[3], groups=256, kernel=(3, 3), stride=(1, 1),
                         padding=(1, 1)),
            ])

        self.conv_sep = ConvBlock(128 * self.scale, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.features = GDC(num_features)
        self._initialize_weights()

    def _initialize_weights(self):
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

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            for func in self.layers:
                # print(func)
                x = func(x)
                # print(x.shape)
        x = self.conv_sep(x.float() if self.fp16 else x)
        x = self.features(x)
        return x


def get_mbf(fp16, num_features, blocks=(1, 2, 6, 2), scale=1):
    return MobileFaceNet(fp16, num_features, blocks, scale=scale)


def get_mbf_large(fp16, num_features, blocks=(2, 8, 12, 4), scale=4):
    return MobileFaceNet(fp16, num_features, blocks=(2, 8, 12, 4), scale=scale)


if __name__ == '__main__':
    import torchsummary

    net = get_mbf(False, 512)
    # # You need to define input size to calcualte parameters
    # torchsummary.summary(net, input_size=(3, 112, 112))

    import torchinfo

    # net = MobileNetV3_Small_Face()
    # You need to define input size to calcualte parameters
    # torchinfo.summary(net, input_size=(3, 112, 112),batch_dim=0)
    # block = Block(20)
    torchinfo.summary(net, input_size=(1, 112, 112), batch_dim=0)
