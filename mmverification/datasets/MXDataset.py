import math
import os
import os.path as osp
import random
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from mmengine.dataset import BaseDataset
from mmengine.fileio import load
from mmengine.utils import is_abs
from torch.utils.data import Dataset

from ..registry import DATASETS

try:
    # 提取出了mxnet的recordio， 避免安装mxnet时候出现的各种版本冲突问题
    # 独立的recordio安装方式详见： https://github.com/terancejiang/MXNet-RecordIO-Standalone
    import mx_recordio as mx
except ImportError:
    # 原版mxnet的recordio
    import mxnet as mx


@DATASETS.register_module()
class MXDataset_palm(Dataset):
    def __init__(self, root_dir, local_rank, im_size=112):
        super(MXDataset_palm, self).__init__()

        self.im_size = im_size

        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')

        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')


        # header, _ = mx.recordio.unpack(s)
        # if header.flag > 0:
        #     self.header0 = (int(header.label[0]), int(header.label[1]))
        #     self.imgidx = np.array(range(1, int(header.label[0])))
        #
        # else:
        self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)

        img = mx.image.imdecode(img).asnumpy()

        if img.shape[0] != self.im_size or img.shape[1] != self.im_size:
            img = cv2.resize(img, (self.im_size, self.im_size))

        img = self.transform(image=img)['image']

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        img = (img.transpose((2, 0, 1)) - 127.5) * 0.0078125
        img = torch.from_numpy(img.astype(np.float32))

        label = header.id
        label = int(label)
        # if not isinstance(label, numbers.Number):
        #     label = label[0]

        label = torch.tensor(label, dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.imgidx)
