_base_ = ["/HOME/scz0bdh/run/mmlab_homework/voc2012/checkpoints/faster_rcnn_r50_fpn_1x_coco.py"]
CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

# # # # #
# Model #
# # # # #
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=len(CLASSES),
        )
    ),
    test_cfg=dict(nms=dict(type="softnms")),
)


# # # # #
# Data  #
# # # # #
dataset_type = 'VOCDataset'
data_root = "/data/public/PascalVOC/2012/VOC2012/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(500, 374), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(500, 374),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    _delete_=True,
    samples_per_gpu=50,
    workers_per_gpu=4,
    train=dict(
        type="RepeatDataset",
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file="/data/public/PascalVOC/2012/VOC2012/ImageSets/Main/train.txt",
            img_prefix="/data/public/PascalVOC/2012/VOC2012/",
            pipeline=train_pipeline,
        )
    ),

    val=dict(
        type=dataset_type,
        ann_file="/data/public/PascalVOC/2012/VOC2012/ImageSets/Main/val.txt",
        img_prefix="/data/public/PascalVOC/2012/VOC2012/",
        pipeline=test_pipeline,
    ),

    test=dict(
        type=dataset_type,
        ann_file="/data/public/PascalVOC/2012/VOC2012/ImageSets/Main/test.txt",
        img_prefix="/data/public/PascalVOC/2012/VOC2012/",
        pipeline=test_pipeline,
    ),
)

# # # # #
# Train #
# # # # #
evaluation = dict(metric=['mAP'], save_best="auto")
optimizer = dict(_delete_=True, type='Adam', lr=2e-3, weight_decay=3e-4)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='exp',
    warmup_iters=5,
    warmup_ratio=0.05,
    warmup_by_epoch=True,
    min_lr=1e-7,
    by_epoch=True,
)
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=6)
log_config = dict(interval=500, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
load_from = "/HOME/scz0bdh/run/mmlab_homework/voc2012/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
workflow = [('train', 8), ("val", 1)]
work_dir = "/HOME/scz0bdh/run/mmlab_homework/voc2012/work_dir"