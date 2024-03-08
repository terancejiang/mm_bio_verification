#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Project Name: mm_bio_verification
File Created: 2024/2/26 下午2:22
Author: Ying.Jiang
File Name: dataset_utils.py
"""
import numpy as np


def rotate_point(pt, rot_mat):
    """
    rotate x,y points by give rotate matrix
    :param pt:
    :param rot_mat:
    :return:
    """
    new_pt = np.array([pt[0], pt[1], 1])
    new_pt = np.dot(rot_mat, new_pt)
    return int(new_pt[0]), int(new_pt[1])