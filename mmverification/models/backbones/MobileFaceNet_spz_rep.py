'''
Adapted from https://github.com/cavalleria/cavaface.pytorch/blob/master/backbone/mobilefacenet.py
Original author cavalleria
'''
import sys

import numpy as np
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module, ReLU, GELU
import torch
sys.path.append('../../')
from recognition.arcface_torch.backbones.DiverseBranchBlock.diversebranchblock import DiverseBranchBlock


class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepLinear(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False,):
        super(RepLinear, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        print(kernel_size)
        assert kernel_size == 1
        assert padding == 1
        padding_11 = padding - kernel_size // 2
        if deploy:

            self.rbr_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                     padding=padding_11, groups=groups, bias=False)
        else:
            self.rbr_reparam_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        print(kernel_size)
        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
            # self.rbr_identity = nn.BatchNorm2d(
            #     num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_identity = None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            # self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
            #                        padding=padding_11, groups=groups)
            self.rbr_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                     padding=padding_11, groups=groups, bias=False)
            print('RepVGG Block, identity = ', self.rbr_identity)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    #   Optional. This may improve the accuracy and facilitates quantization in some cases.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()
    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1,
                                                                                                                   1, 1,
                                                                                                                   1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1,
                                                                                                             1).detach()

        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2,
                                            1:2] ** 2).sum()  # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1  # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (
                    t3 ** 2 + t1 ** 2)).sum()  # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle

    #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    #   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
    #   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class ConvBlock(Module):
    def __init__(self, in_c, out_c, kernel=1, stride=1, padding=(0, 0), groups=1):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            # Conv2d(in_c, out_c, kernel, groups=groups, stride=stride, padding=padding, bias=False),
            RepVGGBlock(in_c, out_c, kernel, stride, padding, groups=groups),

            BatchNorm2d(num_features=out_c),
            # PReLU(num_parameters=out_c)
            # ReLU()
            # GELU()
        )

    def forward(self, x):
        return self.layers(x)


class LinearBlock(Module):
    def __init__(self, in_c, out_c, kernel=1, stride=1, padding=(0, 0), groups=1):
        super(LinearBlock, self).__init__()
        self.layers = nn.Sequential(
            # Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
            RepVGGBlock(in_c, out_c, kernel, stride, padding, groups=groups),
            BatchNorm2d(num_features=out_c)
        )

    def forward(self, x):
        return self.layers(x)

class Linear_block_DBB(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block_DBB, self).__init__()
        self.some_dbb = DiverseBranchBlock(in_channels=in_c, out_channels=out_c, kernel_size=kernel[0],
                                           stride=stride[0], padding=padding[0],
                                           groups=groups, deploy=False)  # 20210902 普通任何卷积改为dbb

    def forward(self, x):
        x = self.some_dbb(x)  # 20210906 改为dbb
        return x

class DepthWise(Module):
    def __init__(self, in_c, out_c, residual=False, kernel=3, stride=2, padding=1, groups=1):
        super(DepthWise, self).__init__()
        self.residual = residual
        self.layers = nn.Sequential(
            ConvBlock(in_c, out_c=groups, kernel=1, padding=(0, 0), stride=1),
            ConvBlock(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride),
            # LinearBlock(groups, out_c, kernel=1, padding=(0, 0), stride=1)
            Linear_block_DBB(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
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
    def __init__(self, c, num_block, groups, kernel=3, stride=1, padding=1):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(DepthWise(c, c, True, kernel, stride, padding, groups))
        self.layers = Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class GDC(Module):
    def __init__(self, embedding_size):
        super(GDC, self).__init__()
        self.layers = nn.Sequential(
            # LinearBlock(512, 512, groups=512, kernel=(7, 7), stride=1, padding=(0, 0)),
            Linear_block_DBB(512, 512, groups=512, kernel=(71, 7), padding=(0, 0), stride=(1, 1)),
            nn.Flatten(),
            Linear(512, embedding_size, bias=False),
            BatchNorm1d(embedding_size))

    def forward(self, x):
        return self.layers(x)


class MobileFaceNet(Module):
    def __init__(self, fp16=False, num_features=512, blocks=(1, 4, 6, 2), scale=1):
        super(MobileFaceNet, self).__init__()
        self.scale = scale
        self.fp16 = fp16
        self.layers = nn.ModuleList()
        self.layers.append(
            ConvBlock(1, 64 * self.scale, kernel=3, stride=2 , padding=1)
        )
        if blocks[0] == 1:
            self.layers.append(
                ConvBlock(64 * self.scale, 64 * self.scale, kernel=3, stride=1, padding=1, groups=64)
            )
        else:
            self.layers.append(
                Residual(64 * self.scale, num_block=blocks[0], groups=128, kernel=3, stride=1,
                         padding=1),
            )

        self.layers.extend(
            [
                DepthWise(64 * self.scale, 64 * self.scale, kernel=3, stride=2, padding=1, groups=128),
                Residual(64 * self.scale, num_block=blocks[1], groups=128, kernel=3, stride=1,
                         padding=1),
                DepthWise(64 * self.scale, 128 * self.scale, kernel=3, stride=2, padding=1, groups=256),
                Residual(128 * self.scale, num_block=blocks[2], groups=256, kernel=3, stride=1,
                         padding=1),
                DepthWise(128 * self.scale, 128 * self.scale, kernel=3, stride=2, padding=1, groups=512),
                Residual(128 * self.scale, num_block=blocks[3], groups=256, kernel=3, stride=1,
                         padding=1),
            ])

        self.conv_sep = ConvBlock(128 * self.scale, 512, kernel=1, stride=1, padding=(0, 0))
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
                x = func(x)
        x = self.conv_sep(x.float() if self.fp16 else x)
        x = self.features(x)
        return x


def get_mbf(fp16, num_features, blocks=(1, 4, 6, 2), scale=1):
    return MobileFaceNet(fp16, num_features, blocks, scale=scale)


def get_mbf_large(fp16, num_features, blocks=(2, 8, 12, 4), scale=4):
    return MobileFaceNet(fp16, num_features, blocks=(2, 8, 12, 4), scale=scale)


if __name__ == '__main__':
    import torchsummary

    net = get_mbf_large(False, 512)
    # # You need to define input size to calcualte parameters
    # torchsummary.summary(net, input_size=(3, 112, 112))

    import torchinfo

    # net = MobileNetV3_Small_Face()
    # You need to define input size to calcualte parameters
    # torchinfo.summary(net, input_size=(3, 112, 112),batch_dim=0)
    # block = Block(20)
    torchinfo.summary(net, input_size=(1, 112, 112), batch_dim=0)
