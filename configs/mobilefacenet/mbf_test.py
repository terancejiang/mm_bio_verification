_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

data_preprocessor = dict(
    type='BioDataPreprocessor',
    mean=[127.5, 127.5, 127.5],
    std=[128., 128., 128.],
    bgr_to_rgb=True,
    pad_size_divisor=32)