metainfo = {
    'dataset_name':'Ear_Keypoint_Dataset_coco',
    'classes':'ear',
    'keypoint_info': {
        0: {'name': '肾上腺', 'id': 0, 'color': [101, 205, 228], 'type': '', 'swap': ''},
        1: {'name': '耳尖', 'id': 1, 'color': [240, 128, 128], 'type': '', 'swap': ''},
        2: {'name': '胃', 'id': 2, 'color': [154, 205, 50], 'type': '', 'swap': ''},
        3: {'name': '眼', 'id': 3, 'color': [34, 139, 34], 'type': '', 'swap': ''},
        4: {'name': '口', 'id': 4, 'color': [139, 0, 0], 'type': '', 'swap': ''},
        5: {'name': '肝', 'id': 5, 'color': [255, 165, 0], 'type': '', 'swap': ''},
        6: {'name': '对屏尖', 'id': 6, 'color': [255, 0, 255], 'type': '', 'swap': ''},
        7: {'name': '心', 'id': 7, 'color': [255, 255, 0], 'type': '', 'swap': ''},
        8: {'name': '肺', 'id': 8, 'color': [29, 123,243], 'type': '', 'swap': ''},
        9: {'name': '肺2', 'id': 9, 'color': [0, 255, 255], 'type': '', 'swap': ''},
        10: {'name': '膀胱', 'id': 10, 'color': [128, 0, 128], 'type': '', 'swap': ''},
        11: {'name': '脾', 'id': 11, 'color': [74, 181, 57], 'type': '', 'swap': ''},
        12: {'name': '角窝中', 'id': 12, 'color': [165, 42, 42], 'type': '', 'swap': ''},
        13: {'name': '神门', 'id': 13, 'color': [128, 128, 0], 'type': '', 'swap': ''},
        14: {'name': '肾', 'id': 14, 'color': [255, 0, 0], 'type': '', 'swap': ''},
        15: {'name': '耳门', 'id': 15, 'color': [34, 139, 34], 'type': '', 'swap': ''},
        16: {'name': '听宫', 'id': 16, 'color': [255, 129, 0], 'type': '', 'swap': ''},
        17: {'name': '听会', 'id': 17, 'color': [70, 130, 180], 'type': '', 'swap': ''},
        18: {'name': '肩', 'id': 18, 'color': [63, 103,165], 'type': '', 'swap': ''},
        19: {'name': '扁桃体', 'id': 19, 'color': [66, 77, 229], 'type': '', 'swap': ''},
        20: {'name': '腰骶椎', 'id': 20, 'color': [255, 105, 180], 'type': '', 'swap': ''}
    }, 
    'skeleton_info': {
        0: {'link':('眼','扁桃体'),'id': 0,'color': [100,150,200]},
        1: {'link':('耳门','听宫'),'id': 1,'color': [200,100,150]},
        2: {'link':('听宫','听会'),'id': 2,'color': [150,120,100]},
        3: {'link':('耳门','听会'),'id': 3,'color': [66,77,229]}
    },
    'joint_weights':[1.0] * 21,
    'sigmas':[0.025] * 21
}
NUM_CLASS = len(metainfo["keypoint_info"])


default_scope = 'mmpose'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='NME',
        rule='less',
        max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=50,
        switch_pipeline=[
            dict(type='LoadImage', backend_args=dict(backend='local')),
            dict(type='GetBBoxCenterScale'),
            dict(type='RandomFlip', direction='horizontal'),
            dict(
                type='RandomBBoxTransform',
                shift_factor=0.0,
                scale_factor=[0.75, 1.25],
                rotate_factor=60),
            dict(type='TopdownAffine', input_size=(256, 256)),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(
                type='Albumentation',
                transforms=[
                    dict(type='Blur', p=0.1),
                    dict(type='MedianBlur', p=0.1),
                    dict(
                        type='CoarseDropout',
                        max_holes=1,
                        max_height=0.4,
                        max_width=0.4,
                        min_holes=1,
                        min_height=0.2,
                        min_width=0.2,
                        p=0.5)
                ]),
            dict(
                type='GenerateTarget',
                encoder=dict(
                    type='SimCCLabel',
                    input_size=(256, 256),
                    sigma=(5.66, 5.66),
                    simcc_split_ratio=2.0,
                    normalize=False,
                    use_dark=False)),
            dict(type='PackPoseInputs')
        ])
]
work_dir = "work_dir"
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)
log_level = 'INFO'
load_from = "checkpoint/rtmpose-m_simcc-coco-wholebody-face_pt-aic-coco_60e-256x256-62026ef2_20230228.pth"
resume = False
backend_args = dict(backend='local')
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=10)
val_cfg = dict()
test_cfg = dict()
stage2_num_epochs = 10
base_lr = 0.004
randomness = dict(seed=21)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.004, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-05, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=30,
        end=60,
        T_max=30,
        by_epoch=True,
        convert_to_iter_based=True)
]
auto_scale_lr = dict(base_batch_size=512)
codec = dict(
    type='SimCCLabel',
    input_size=(256, 256),
    sigma=(5.66, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)


model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint=
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth'
        )),
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=NUM_CLASS,
        input_size=(256, 256),
        in_featuremap_size=(8, 8),
        simcc_split_ratio=2.0,
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.0,
            label_softmax=True),
        decoder=dict(
            type='SimCCLabel',
            input_size=(256, 256),
            sigma=(5.66, 5.66),
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False)),
    test_cfg=dict(flip_test=True))


data_type="BaseCocoStyleDataset"
data_root='Ear210_Keypoint_Dataset_coco/'
train_dataloader = dict(
    batch_size=64,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=data_type,
        data_root=data_root,
        data_mode='topdown',
        ann_file='train_coco.json',
        data_prefix=dict(img='images/'),
        metainfo=metainfo,
        pipeline=[
            dict(type='LoadImage', backend_args=dict(backend='local')),
            dict(type='GetBBoxCenterScale'),
            dict(type='RandomFlip', direction='horizontal'),
            dict(
                type='RandomBBoxTransform',
                scale_factor=[0.6, 1.4],
                rotate_factor=80),
            dict(type='TopdownAffine', input_size=(256, 256)),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(
                type='Albumentation',
                transforms=[
                    dict(type='Blur', p=0.1),
                    dict(type='MedianBlur', p=0.1),
                    dict(
                        type='CoarseDropout',
                        max_holes=1,
                        max_height=0.4,
                        max_width=0.4,
                        min_holes=1,
                        min_height=0.2,
                        min_width=0.2,
                        p=1.0)
                ]),
            dict(
                type='GenerateTarget',
                encoder=dict(
                    type='SimCCLabel',
                    input_size=(256, 256),
                    sigma=(5.66, 5.66),
                    simcc_split_ratio=2.0,
                    normalize=False,
                    use_dark=False)),
            dict(type='PackPoseInputs')
        ]))
val_dataloader = dict(
    batch_size=64,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=data_type,
        data_root=data_root,
        data_mode='topdown',
        ann_file='val_coco.json',
        data_prefix=dict(img='images/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=[
            dict(type='LoadImage', backend_args=dict(backend='local')),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(256, 256)),
            dict(type='PackPoseInputs')
        ]))
test_dataloader = val_dataloader

val_evaluator = [
    dict(type='CocoMetric', ann_file=data_root + 'val_coco.json'),
    dict(type='PCKAccuracy'),
    dict(type='AUC'),
    dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=[1, 2])
]
test_evaluator = val_evaluator
