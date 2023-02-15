_base_ = ["./checkpoint/deeplabv3plus_r50-d8_512x512_80k_ade20k.py"]


CLASSES = [
    "Background (waterbody)", "Human divers", "Aquatic plants and sea-grass",
    "Wrecks and ruins", "Robots (AUVs/ROVs/instruments)", "Reefs and invertebrates",
    "Fish and vertebrates", "Sea-floor and rocks"
]
PALETTE = [(0,0,0), (0,0,255), (0,255,0), (0,255,255), (255,0,0), (255,0,255), (255,255,0), (255,255,255)]
# # # # #
# Model #
# # # # #
model = dict(
    # backbone=dict(
    #     norm_cfg=dict(type='BN', requires_grad=True),
    # ),
    decode_head=dict(
        # norm_cfg=dict(type='BN', requires_grad=True),
        num_classes=len(CLASSES),
        # loss_decode=dict(_delete_=True, type="DiceLoss", loss_weight=1.),
    ),
    auxiliary_head=dict(
        # norm_cfg=dict(type='BN', requires_grad=True),
        num_classes=len(CLASSES),
        # loss_decode=dict(_delete_=True, type="DiceLoss", loss_weight=.4),
    )
)

# # # # # #
# Dataset #
# # # # # #
dataset_type = "CustomDataset"
data_root = "./data/"
img_suffix = ".jpg"
mask_suffix = ".png"
img_dir = "images"
ann_dir = "pre_mask"
train_split = "train.txt"
val_split = "val.txt"
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
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
        ])
]

data = dict(
    _delete_=True,
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        split=train_split,
        classes=CLASSES,
        palette=PALETTE,
        img_suffix=img_suffix,
        seg_map_suffix=mask_suffix,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        split=val_split,
        classes=CLASSES,
        palette=PALETTE,
        img_suffix=img_suffix,
        seg_map_suffix=mask_suffix,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        split=val_split,
        classes=CLASSES,
        palette=PALETTE,
        img_suffix=img_suffix,
        seg_map_suffix=mask_suffix,
        pipeline=test_pipeline
    ),
)


# # # # #
# Train #
# # # # #
evaluation = dict(by_epoch=True, interval=2, metric=['mIoU'], save_best="auto", pre_eval=True)
optimizer = dict(_delete_=True, type='Adam', lr=5e-3, weight_decay=1e-4)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    min_lr=0,
    by_epoch=False,
)
runner = dict(_delete_=True, type='EpochBasedRunner', max_epochs=80)
# runner = dict(_delete_=True, type='IterBasedRunner', max_iters=800)
checkpoint_config = dict(interval=1, by_epoch=True)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
load_from = "./checkpoint/deeplabv3plus_r50-d8_512x512_80k_ade20k_20200614_185028-bf1400d8.pth"
workflow = [('train', 1)]
work_dir = "work_dir"
gpu_ids = [0, 1]