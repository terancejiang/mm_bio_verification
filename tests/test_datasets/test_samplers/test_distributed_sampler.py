#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Project Name: mm_bio_verification
File Created: 2024/2/26 下午1:42
Author: Ying.Jiang
File Name: test_distributed_sampler.py
"""
import unittest

import torch
from torch.utils.data import Dataset

from mmverification.datasets.samplers.pytorch_distributed import DistributedSampler


class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # Return (index, value) pair, where value is a random tensor
        return index, torch.rand(1)


class TestDistributedSampler(unittest.TestCase):
    def test_distributed_sampler(self):
        dataset = DummyDataset(size=100)
        num_replicas = 4  # Simulating 4 distributed processes
        rank = 0  # Test rank 0 to start with

        sampler = DistributedSampler(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=True,
            seed=42
        )

        # Generate indices for rank 0
        indices_for_rank_0 = list(iter(sampler))

        # Assert that the sampled indices are correct and non-overlapping
        self.assertEqual(len(indices_for_rank_0), len(dataset) // num_replicas)
        self.assertTrue(all(i < len(dataset) for i in indices_for_rank_0))


if __name__ == '__main__':
    unittest.main()
