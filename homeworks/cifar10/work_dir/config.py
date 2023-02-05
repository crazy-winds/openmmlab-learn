model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        frozen_stages=4),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))
dataset_type = 'CIFAR10'
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[125.307, 122.961, 113.8575],
        std=[51.5865, 50.847, 51.255],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(
        type='Normalize',
        mean=[125.307, 122.961, 113.8575],
        std=[51.5865, 50.847, 51.255],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=2,
    train=dict(
        type='CIFAR10',
        data_prefix='data/CIFAR-10',
        pipeline=[
            dict(type='RandomCrop', size=32, padding=4),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[125.307, 122.961, 113.8575],
                std=[51.5865, 50.847, 51.255],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='CIFAR10',
        data_prefix='data/CIFAR-10',
        pipeline=[
            dict(
                type='Normalize',
                mean=[125.307, 122.961, 113.8575],
                std=[51.5865, 50.847, 51.255],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True),
    test=dict(
        type='CIFAR10',
        data_prefix='data/cifar10',
        pipeline=[
            dict(
                type='Normalize',
                mean=[125.307, 122.961, 113.8575],
                std=[51.5865, 50.847, 51.255],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True))
optimizer = dict(
    type='SGD',
    lr=7e-05,
    momentum=0.9,
    weight_decay=0.0003,
    paramwise_cfg=dict(norm_decay_mult=0))
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=1)
log_config = dict(interval=150, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = './checkpoints/resnet18_b16x8_cifar10_20210528-bd6371c8.pth'
resume_from = None
workflow = [('train', 4), ('val', 1)]
evaluation = dict(
    interval=5,
    metric='accuracy',
    metric_options=dict(topk=(1, )),
    save_best='auto')
work_dir = 'work_dir'
gpu_ids = [0]
