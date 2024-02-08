data_preprocessor = dict(
    mean=[
        151.78,
        103.29,
        97.41,
    ], std=[
        69.92,
        55.96,
        54.84,
    ], to_rgb=True)
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
load_from = None
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
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        num_classes=4,
        topk=1,
        type='LinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(lr=0.1, momentum=0.9, type='SGD', weight_decay=0.0001))
param_scheduler = dict(
    by_epoch=False, gamma=0.1, milestones=[
        30,
        60,
        90,
    ], type='MultiStepLR')
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=1,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='meta/train.txt',
        classes=[
            'normal',
            'polyps',
            'esophagitis',
            'barretts',
        ],
        data_prefix='train',
        data_root='../../images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(interpolation='bicubic', scale=(
                320,
                320,
            ), type='Resize'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset',
        with_label=True),
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(topk=1, type='Accuracy'),
    dict(
        average=None,
        items=[
            'precision',
            'recall',
            'f1-score',
        ],
        type='SingleLabelMetric'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(interpolation='bicubic', scale=(
        320,
        320,
    ), type='Resize'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
train_dataloader = dict(
    batch_size=4,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='meta/train.txt',
        classes=[
            'normal',
            'polyps',
            'esophagitis',
            'barretts',
        ],
        data_prefix='train',
        data_root='../../images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(interpolation='bicubic', scale=(
                320,
                320,
            ), type='Resize'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset',
        with_label=True),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(num_classes=4, shuffle=True, type='DynamicSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(interpolation='bicubic', scale=(
        320,
        320,
    ), type='Resize'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=1,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='meta/train.txt',
        classes=[
            'normal',
            'polyps',
            'esophagitis',
            'barretts',
        ],
        data_prefix='train',
        data_root='../../images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(interpolation='bicubic', scale=(
                320,
                320,
            ), type='Resize'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset',
        with_label=True),
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(topk=1, type='Accuracy'),
    dict(
        average=None,
        items=[
            'precision',
            'recall',
            'f1-score',
        ],
        type='SingleLabelMetric'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
work_dir = '../mmpretrain/tools'
