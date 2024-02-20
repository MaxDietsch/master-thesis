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
    checkpoint=dict(interval=5, max_keep_ckpts=5, type='CheckpointHook'),
    logger=dict(interval=2014, type='LoggerHook'),
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
    backbone=dict(arch='121', type='DenseNet'),
    head=dict(
        in_channels=1024,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        num_classes=4,
        topk=1,
        type='CoSenLinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='CoSenClassifier')
optim_wrapper = dict(
    optimizer=dict(lr=1e-05, momentum=0.9, type='SGD', weight_decay=0.0001))
param_scheduler = dict(
    by_epoch=True, gamma=0.1, milestones=[
        100,
        200,
        300,
    ], type='MultiStepLR')
resume = False
s_freq = 1
s_samples_per_class = [
    10,
    10,
    10,
    10,
]
samples_per_class = [
    150,
    45,
    132,
    539,
]
test_cfg = dict()
test_dataloader = dict(
    batch_size=4,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='meta/val.txt',
        classes=[
            'normal',
            'polyps',
            'barretts',
            'esophagitis',
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
    num_workers=4,
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
        640,
        640,
    ), type='Resize'),
    dict(type='PackInputs'),
]
train_cfg = dict(
    by_epoch=3,
    max_epochs=400,
    s_freq=1,
    s_samples_per_class=[
        10,
        10,
        10,
        10,
    ],
    samples_per_class=[
        150,
        45,
        132,
        539,
    ],
    val_interval=1)
train_dataloader = dict(
    batch_size=4,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='meta_min/train.txt',
        classes=[
            'normal',
            'polyps',
            'barretts',
            'esophagitis',
        ],
        data_prefix='train',
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
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(interpolation='bicubic', scale=(
        640,
        640,
    ), type='Resize'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=4,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='meta/val.txt',
        classes=[
            'normal',
            'polyps',
            'barretts',
            'esophagitis',
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
    num_workers=4,
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
work_dir = './work_dirs/cosen_densenet'
