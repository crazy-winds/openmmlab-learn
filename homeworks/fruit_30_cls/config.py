import os
_base_ = [
    "checkpoint/resnet50_8xb32_in1k.py"
]

metainfo = {"classes": sorted(os.listdir("fruit_dataset/"))}
num_classes = len(os.listdir("fruit_dataset/"))

# # # # #
# Model #
# # # # #
model = dict(
    backbone=dict(
        frozen_stages=1
    ),
    head=dict(
        num_classes=num_classes,
        topk=(1, )
    )
)

# # # # 
# Data #
# # # # 
data_preprocessor = dict(
    num_classes=num_classes,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_dataloader = dict(
    dataset=dict(
        data_root="fruit_dataset/",
        ann_file="",
        data_prefix="",
        metainfo=metainfo
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root="fruit_dataset/",
        ann_file="",
        data_prefix="",
        metainfo=metainfo
    )
)

test_dataloader = dict(
    dataset=dict(
        data_root="fruit_dataset/",
        ann_file="",
        data_prefix="",
        metainfo=metainfo
    )
)

# # # # # #
# Runtime #
# # # # # #
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[10, 20, 30], gamma=0.5)
train_cfg = dict(by_epoch=True, max_epochs=40, val_interval=1)

val_evaluator = dict(type='Accuracy', topk=(1, ))
default_hooks=dict(
    checkpoint = dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=5,
        save_best="auto"
    )
)

load_from = "checkpoint/resnet50_8xb32_in1k_20210831-ea4938fc.pth"
work_dir = "work_dir"