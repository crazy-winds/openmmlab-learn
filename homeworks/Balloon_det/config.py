_base_ = [
    "checkpoint/rtmdet_tiny_8xb32-300e_coco.py"
]

metainfo = {
    "classes": ("balloon", ),
    "palette": [(255, 192, 203)]
}

# # # # # 
# Model #
# # # # # 
model = dict(
    bbox_head=dict(
        num_classes=1,
    )
)

# # # # # #
# Dataset #
# # # # # #
train_dataloader = dict(
    batch_size=8,
    num_workers=5,
    dataset=dict(
        data_root="./data/balloon/train/",
        ann_file="coco_train.json",
        data_prefix=dict(img=""),
        metainfo=metainfo
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root="./data/balloon/val/",
        ann_file="coco_val.json",
        data_prefix=dict(img=""),
        metainfo=metainfo
    )
)

test_dataloader = val_dataloader

# # # # # # 
# Runtime #
# # # # # # 
stage2_num_epochs = 5
val_evaluator = dict(ann_file='./data/balloon/val/coco_val.json')
test_evaluator = val_evaluator
train_cfg = dict(
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
    optimizer=dict(type='AdamW', lr=3e-4, weight_decay=3e-4),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

load_from = "checkpoint/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
work_dir = "work_dir"
