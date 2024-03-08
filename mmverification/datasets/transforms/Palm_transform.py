#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Project Name: mm_bio_verification
File Created: 2024/2/22 下午2:30
Author: Ying.Jiang
File Name: Palm_transform.py
"""

import albumentations as A

from mmcv.transforms import BaseTransform, TRANSFORMS


@TRANSFORMS.register_module()
class PalmTransform(BaseTransform):
    def __init__(self, p1=0.4, p2=0.5, p3=0.2):
        super().__init__()
        self.transform = A.Compose([
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.ISONoise(intensity=(0.15, 1.3), p=0.5),
            ], p=0.1),
            A.OneOf([
                A.Rotate(limit=(-5, 5), p=1),
                # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=5, p=1),
            ], p=0.1),
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1),  # 动态模糊
                A.MedianBlur(blur_limit=3, p=0.05),  # 中值模糊
                A.ImageCompression(quality_lower=30, quality_upper=70, p=1),
                A.GlassBlur(sigma=0.8, max_delta=1, iterations=1, mode="fast", p=1),
                A.GaussianBlur(blur_limit=(3, 9), sigma_limit=0, p=1),
            ], p=0.1),
            A.OneOf(
                [
                    A.Sharpen(alpha=(0.0, 0.2), lightness=(0.1, 1), p=1),  # 锐化
                    A.RandomBrightnessContrast(brightness_limit=(-0.15, 0.1), contrast_limit=(-0.2, 0.2), p=1),  # 亮度对比度
                    A.RandomGamma(gamma_limit=(100, 180), p=1)
                ], p=0.1),
        ], p=p1)

    def transform(self, results: dict) -> dict:
        img = results['img']
        results['img'] = self.transform(image=img)['image']
        return results


