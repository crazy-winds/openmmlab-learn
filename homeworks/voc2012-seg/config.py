_base_ = ["./checkpoint/deeplabv3_r50-d8_512x512_20k_voc12aug.py"]
CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor')

# # # # #
# Model #
# # # # #
model = dict(
    backbone=dict(
        norm_cfg=dict(type='BN', requires_grad=True),
    ),
    decode_head=dict(
        norm_cfg=dict(type='BN', requires_grad=True),
        num_classes=len(CLASSES),
        # loss_decode=dict(_delete_=True, type="FocalLoss", loss_weight=1.),
    ),
    auxiliary_head=dict(
        norm_cfg=dict(type='BN', requires_grad=True),
        num_classes=len(CLASSES),
        # loss_decode=dict(_delete_=True, type="FocalLoss", loss_weight=.4),
    )
)

# # # # # #
# Dataset #
# # # # # #
dataset_type = "PascalVOCDataset"
data_root = "/data/public/PascalVOC/2012/VOC2012/"
img_suffix = ".jpg"
mask_suffix = ".png"
img_dir = "JPEGImages"
ann_dir = "SegmentationClass"
train_split = "ImageSets/Segmentation/train.txt"
val_split = "ImageSets/Segmentation/val.txt"
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(640, 320), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(320, 320), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(320, 320), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 320),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]
    )
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        split=train_split,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        split=val_split,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        split=val_split,
        pipeline=test_pipeline,
    ),
)

# # # # #
# Train #
# # # # #
evaluation = dict(by_epoch=True, interval=2, metric=['mIoU'], save_best="auto", pre_eval=True)
optimizer = dict(_delete_=True, type='Adam', lr=1e-4, weight_decay=1e-4)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    min_lr=0,
    by_epoch=False,
)
runner = dict(_delete_=True, type='EpochBasedRunner', max_epochs=10)
checkpoint_config = dict(interval=1, by_epoch=True)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
load_from = "./checkpoint/deeplabv3_r50-d8_512x512_20k_voc12aug_20200617_010906-596905ef.pth"
workflow = [('train', 1)]
work_dir = "work_dir"