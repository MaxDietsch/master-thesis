data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
dataset_type = 'CustomDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = '../work_dirs/resnet50_sgd_bepnwom_default_cw/epoch_3.pth'
log_level = 'INFO'
model = dict(
    backbone=dict(
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        type='ResNet'),
    head=dict(
        in_channels=2048,
        loss=dict(
            class_weight=[
                1.0,
                10.0,
                10.0,
                2.0,
            ],
            loss_weight=1.0,
            type='CrossEntropyLoss'),
        num_classes=4,
        topk=1,
        type='LinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(lr=0.1, momentum=0.9, type='SGD', weight_decay=0.0001))
param_scheduler = dict(
    by_epoch=True, gamma=0.1, milestones=[
        30,
        60,
        90,
    ], type='MultiStepLR')
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='meta/val.txt',
        classes=[
            'normal',
            'polyps',
            'esophagitis',
            'barretts',
        ],
        data_prefix='val',
        data_root='../../B_E_P_N-without-mix',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(interpolation='bicubic', scale=(
                640,
                640,
            ), type='Resize'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset',
        with_label=True),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(topk=1, type='Accuracy')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(interpolation='bicubic', scale=(
        640,
        640,
    ), type='Resize'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='meta/train.txt',
        classes=[
            'normal',
            'polyps',
            'esophagitis',
            'barretts',
        ],
        data_prefix='train',
        data_root='../../B_E_P_N-without-mix',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(interpolation='bicubic', scale=(
                640,
                640,
            ), type='Resize'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset',
        with_label=True),
    num_workers=5,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(interpolation='bicubic', scale=(
        640,
        640,
    ), type='Resize'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='meta/val.txt',
        classes=[
            'normal',
            'polyps',
            'esophagitis',
            'barretts',
        ],
        data_prefix='val',
        data_root='../../B_E_P_N-without-mix',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(interpolation='bicubic', scale=(
                640,
                640,
            ), type='Resize'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset',
        with_label=True),
    num_workers=5,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(topk=1, type='Accuracy')
vis_backends = [
    dict(type='LocalVisBackend'),
]
work_dir = './work_dirs/resnet50_sgd_bepnwom_default'
