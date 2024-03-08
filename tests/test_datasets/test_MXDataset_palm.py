#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Project Name: mm_bio_verification
File Created: 2024/2/22 下午2:21
Author: Ying.Jiang
File Name: test_MXDataset_palm.py
"""
# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmverification.datasets import MXDataset_palm


class TestCocoDataset(unittest.TestCase):

    def test_coco_dataset(self):
        # test CocoDataset
        metainfo = dict(classes=('bus', 'car'), task_name='new_task')
        dataset = MXDataset_palm(
            data_prefix=dict(img='imgs'),
            ann_file='tests/data/coco_sample.json',
            metainfo=metainfo,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[],
            serialize_data=False,
            lazy_init=False)
        self.assertEqual(dataset.metainfo['classes'], ('bus', 'car'))
        self.assertEqual(dataset.metainfo['task_name'], 'new_task')
        self.assertListEqual(dataset.get_cat_ids(0), [0, 1])

    def test_coco_dataset_without_filter_cfg(self):
        # test CocoDataset without filter_cfg
        dataset = CocoDataset(
            data_prefix=dict(img='imgs'),
            ann_file='tests/data/coco_sample.json',
            pipeline=[])
        self.assertEqual(len(dataset), 4)

        # test with test_mode = True
        dataset = CocoDataset(
            data_prefix=dict(img='imgs'),
            ann_file='tests/data/coco_sample.json',
            test_mode=True,
            pipeline=[])
        self.assertEqual(len(dataset), 4)

    def test_coco_annotation_ids_unique(self):
        # test annotation ids not unique error
        metainfo = dict(classes=('car', ), task_name='new_task')
        with self.assertRaisesRegex(AssertionError, 'are not unique!'):
            CocoDataset(
                data_prefix=dict(img='imgs'),
                ann_file='tests/data/coco_wrong_format_sample.json',
                metainfo=metainfo,
                pipeline=[])
