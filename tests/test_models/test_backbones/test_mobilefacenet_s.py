# Copyright (c) OpenMMLab. All rights reserved.
import torchinfo

from mmverification.models.backbones.mobilefacenet import MobileFaceNet
from .utils import check_norm_state


def test_MobileFaceNet_backbone():

    # Test MobileNetV2 with first stage frozen
    model = MobileFaceNet()
    model.train()

    # Test MobileNetV2 with norm_eval=True
    model = MobileFaceNet()
    model.train()

    assert check_norm_state(model.modules(), True)

    # Test MobileNetV2 forward with widen_factor=1.0
    model = MobileFaceNet(scale=2)
    model.eval()

    assert check_norm_state(model.modules(), False)


    torchinfo.summary(model, input_size=(1, 112, 112), batch_dim=0)
