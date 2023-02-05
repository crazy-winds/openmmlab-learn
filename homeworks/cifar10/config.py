_base_ = ["./checkpoints/resnet18_8xb16_cifar10.py"]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(frozen_stages=4),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

dataset_type = 'CIFAR10'
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
        type=dataset_type, data_prefix='data/CIFAR-10', pipeline=train_pipeline),
    val=dict(
        type=dataset_type, data_prefix='data/CIFAR-10', pipeline=test_pipeline),)

evaluation = dict(
    interval=5, metric='accuracy', metric_options={'topk': (1, )}, save_best="auto"
)
optimizer = dict(
    _delete_=True,
    type='SGD',
    lr=7e-5,
    momentum=0.9,
    weight_decay=3e-04,
    paramwise_cfg=dict(norm_decay_mult=0)
)
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    min_lr=0,
    by_epoch=False,
)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=150,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
runner = dict(type='EpochBasedRunner', max_epochs=20)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dir'
load_from = "./checkpoints/resnet18_b16x8_cifar10_20210528-bd6371c8.pth"
resume_from = None
workflow = [('train', 4), ("val", 1)]
