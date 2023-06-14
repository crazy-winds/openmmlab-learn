_base_ = ["checkpoint/pspnet_r18-d8_4xb2-80k_cityscapes-512x1024.py"]


NUM_CLASSES = 6
metainfo = dict(
    classes=("red", "green", "white", "seed-black", "seed-white"),
    palette=[
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3]
    ]
)

# # # # #
# Model #
# # # # #
model = dict(
    decode_head=dict(
        num_classes=NUM_CLASSES
    ),
    auxiliary_head=dict(
        num_classes=NUM_CLASSES
    ),
)

# # # # # # # #
# DataLoader #
# # # # # # #
train_pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(1024, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(384, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

val_pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    sampler=dict(type="DefaultSampler"),
    dataset=dict(
        type="BaseSegDataset",
        data_root="data/Watermelon87_Semantic_Seg_Mask/",
        data_prefix=dict(
            img_path='img_dir/train/',
            seg_map_path='ann_dir/train/'
        ),
        metainfo=metainfo,
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    sampler=dict(type="DefaultSampler"),
    dataset=dict(
        type="BaseSegDataset",
        data_root="data/Watermelon87_Semantic_Seg_Mask/",
        data_prefix=dict(
            img_path='img_dir/val/',
            seg_map_path='ann_dir/val/'
        ),
        metainfo=metainfo,
        pipeline=val_pipeline
    )
)
# # # # # #
# Runtime #
# # # # # #
stage2_num_epochs = 5
# train_cfg = dict(type='IterBasedTrainLoop', max_iters=8000, val_interval=800)
train_cfg = dict(
    _delete_=True,
    type="EpochBasedTrainLoop",
    max_epochs=50,
    val_interval=5,
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=10
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-5,
        begin=10,
        end=50,
        T_max=50,
        by_epoch=True,
        convert_to_iter_based=True
    )
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(_delete_=True, type='AdamW', lr=3e-4, weight_decay=3e-4),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=10, max_keep_ckpts=3),
)

load_from = "checkpoint/pspnet_r18-d8_512x1024_80k_cityscapes_20201225_021458-09ffa746.pth"
work_dir = "work_dir"
