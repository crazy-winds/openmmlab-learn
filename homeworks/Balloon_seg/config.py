_base_ = ["/HOME/scz0bdh/run/mmlab_homework/Balloon/checkpoints/mask_rcnn_r50_fpn_1x_coco.py"]


model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,
        ),
        mask_head=dict(
            num_classes=1,
        )
    )
)

data = dict(
    train=dict(
        ann_file="/HOME/scz0bdh/run/mmlab_homework/Balloon/data/balloon/train/coco_train.json",
        img_prefix="/HOME/scz0bdh/run/mmlab_homework/Balloon/data/balloon/train/",
        classes=("balloon",)
    ),
    val=dict(
        ann_file="/HOME/scz0bdh/run/mmlab_homework/Balloon/data/balloon/val/coco_val.json",
        img_prefix="/HOME/scz0bdh/run/mmlab_homework/Balloon/data/balloon/val/",
        classes=("balloon",)
    )
)

evaluation = dict(metric=['bbox', 'segm'], save_best="auto")
optimizer = dict(_delete_=True, type='Adam', lr=3e-4, weight_decay=1e-4)
optimizer_config = dict(grad_clip=None)
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
load_from = "./checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth"
workflow = [('train', 2), ("val", 1)]
work_dir = "work_dir"