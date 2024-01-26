"""
@author: Jun Wang 
@date: 20201019
@contact: jun21wangustc@gmail.com
"""

# based on:
# https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/model.py

from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, \
    PReLU, Sequential, Module, ReLU, ReLU6, LeakyReLU, GELU
import torch

import torch
import sys

sys.path.append('../../')
from recognition.arcface_torch.backbones.DiverseBranchBlock.diversebranchblock import DiverseBranchBlock


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class Conv_block_DBB(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block_DBB, self).__init__()
        self.some_dbb = DiverseBranchBlock(in_channels=in_c, out_channels=out_c, kernel_size=kernel[0],
                                           stride=stride[0], padding=padding[0],
                                           groups=groups, deploy=False)  # 20210902 普通任何卷积改为dbb
        # self.act = PReLU(out_c)
        self.act = ReLU(out_c)
        # self.act = LeakyReLU(out_c)
        # self.act = GELU()

    def forward(self, x):
        x = self.some_dbb(x)  # 20210902 改为dbb
        x = self.act(x)
        return x


class Linear_block_DBB(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block_DBB, self).__init__()
        self.some_dbb = DiverseBranchBlock(in_channels=in_c, out_channels=out_c, kernel_size=kernel[0],
                                           stride=stride[0], padding=padding[0],
                                           groups=groups, deploy=False)  # 20210902 普通任何卷积改为dbb

    def forward(self, x):
        x = self.some_dbb(x)  # 20210906 改为dbb
        return x


class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding,
                           bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block_DBB(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block_DBB(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block_DBB(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
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
            modules.append(
                Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class MobileFaceNets_DBB(Module):
    def __init__(self, embedding_size=512, out_h=7, out_w=7, scale=1,fp16=True):
        super(MobileFaceNets_DBB, self).__init__()
        self.scale = scale
        self.fp16 = fp16
        self.conv1 = Conv_block_DBB(1, 64*self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block_DBB(64*self.scale, 64*self.scale, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Depth_Wise(64*self.scale, 64*self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64*self.scale, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(64*self.scale, 128*self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128*self.scale, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128*self.scale, 128*self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128*self.scale, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block_DBB(128*self.scale, 512*self.scale, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        # self.conv_6_dw = Linear_block(512, 128, groups=128, kernel=(7,7), stride=(1, 1), padding=(0, 0))
        # self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(4,7), stride=(1, 1), padding=(0, 0)) //半人脸
        self.conv_6_dw = Linear_block(512*self.scale, 512*self.scale, groups=512, kernel=(out_h, out_w), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512*self.scale, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
        # print("*********MobilefaceNet_DBB",x.shape)
            out = self.conv1(x)
            # print("*********conv1 shape ",out.shape)
            out = self.conv2_dw(out)
            # print("*********conv2_dw shape ",out.shape)
            out = self.conv_23(out)
            # print("*********conv_23 shape ",out.shape)
            out = self.conv_3(out)
            # print("*********conv_3 shape ",out.shape)
            out = self.conv_34(out)
            # print("*********conv_34 shape ",out.shape)
            out = self.conv_4(out)
            # print("*********conv_4 shape ",out.shape)
            out = self.conv_45(out)
            # print("*********conv_45 shape ",out.shape)
            out = self.conv_5(out)
            # print("*********conv_5 shape ",out.shape)
            out = self.conv_6_sep(out)
            # print("*********conv_6_sep shape ",out.shape)

        out = self.conv_6_dw(out.float() if self.fp16 else out)
        # out = self.conv_6_dw(out)
            # print("*********conv_6_dw shape ",out.shape)
        out = self.conv_6_flatten(out)

        out = self.linear(out)
        out = self.bn(out)
        return l2_norm(out)
        # return out


if __name__ == "__main__":
    from torch.optim import SGD
    from torch.cuda import amp
    import torch.nn as nn
    # from utils.torch_utils import select_device
    model = MobileFaceNets_DBB(512, 7, 7)

    from torchinfo import summary
    from torch.autograd import Variable

    summary(model, input_size=(1, 1, 112, 112))

    for m in model.modules():
        if hasattr(m, 'switch_to_deploy'):
            m.switch_to_deploy()
    summary(model, input_size=(1, 1, 112, 112))


    # import torch
    #
    # input = torch.Tensor(2, 1, 112, 112)
    #
    #
    # x = torch.randn(2, 1, 112, 112, requires_grad=True).to('cuda')
    #
    # # from torchsummary import summary
    # #
    # # summary(model, (1, 112, 112))
    #
    # from torchinfo import summary
    #
    # summary(model, input_size=(2, 1, 112, 112))

    # from pthflops import count_ops

    device = 'cpu'

    # model = model.to(device)
    # inp = torch.rand(2, 1, 112, 112)

    # Count the number of FLOPs
    # count_ops(model, inp)
    # device = select_device('cuda:0,1', batch_size=16)
    # cuda = device.type
    # optimizer = SGD(net.parameters(), lr=0.01, momentum=0.99, nesterov=True)
    # optimizer.zero_grad()
    # scaler = amp.GradScaler(enabled=cuda)
    # net.to(device)
    #
    # criterion = nn.CrossEntropyLoss().cuda(device)
    # # summary(net, (3,112,112))
    # sample = torch.rand(2,1,112, 112).to(device)
    # label = torch.rand(2,512).to(device)
    #
    # net.train()
    #
    # with amp.autocast():
    #     res = net(sample)
    #     loss =  criterion(res, label)
    # scaler.scale(loss).backward()
    # scaler.step(optimizer)
    # scaler.update()
    # # forward
    # # backward
    #
    #
    # print(loss)
    # print(res)
    # print(label)