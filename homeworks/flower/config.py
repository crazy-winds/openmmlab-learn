_base_ = ["./checkpoints/shufflenet-v2-1x_16xb64_in1k.py"]

model = dict(
    type="ImageClassifier",
    backbone=dict(
        frozen_stages=3,
    ),
    head=dict(
        type="LinearClsHead",
        num_classes=5,
        topk=(1, )
    )
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=196, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=196),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    _delete_=True,
    samples_per_gpu=256,
    workers_per_gpu=2,
    train=dict(
        type="CustomDataset",
        data_prefix="/HOME/scz0bdh/run/mmlab_homework/one/dataset/flower_dataset/train",
        pipeline=train_pipeline,
    ),
    val=dict(
        type="CustomDataset",
        data_prefix="/HOME/scz0bdh/run/mmlab_homework/one/dataset/flower_dataset/val",
        pipeline=test_pipeline,
    )
)

evaluation = dict(metric_options={'topk': (1, )}, save_best="auto")
optimizer = dict(
    _delete_=True,
    type='Adam',
    lr=3e-4,
    weight_decay=4e-05,
    paramwise_cfg=dict(norm_decay_mult=0)
)
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    min_lr=0,
    by_epoch=False,
)

runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
load_from = "/HOME/scz0bdh/run/mmlab_homework/one/checkpoints/shufflenet_v2_batch1024_imagenet_20200812-5bf4721e.pth"
resume_from = None
workflow = [('train', 1), ("val", 1)]
work_dir = "work_dir"