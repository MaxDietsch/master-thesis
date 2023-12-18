#onstruct dataloader and evaluator
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    # Input image data channels in 'RGB' order
    mean=[150.60, 102.57, 97.12],
    std=[70.70, 56.70, 55.68],
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),     # read image
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='Resize', scale=(640, 640), interpolation='bicubic'),
    dict(type='PackInputs'),         # prepare images and labels
]

test_pipeline = [
    dict(type='LoadImageFromFile'),     # read image
    dict(type='Resize', scale=(640, 640), interpolation='bicubic'),
    dict(type='PackInputs'),                 # prepare images and labels
]

train_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='../../rBP_B_E_P_N-without-mix',
        ann_file='meta/train.txt',
        data_prefix='train',
        with_label=True,
        classes=['normal', 'polyps', 'barretts', 'esophagits'],
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='../../rBP_B_E_P_N-without-mix',
        ann_file='meta/val.txt',
        data_prefix='val',
        with_label=True,
        classes=['normal', 'polyps', 'barretts', 'esophagitis'],
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
val_evaluator = [
            dict(type='Accuracy', topk=(1)), 
            dict(type='SingleLabelMetric', items=['precision', 'recall'], average=None)
            ]

test_dataloader = val_dataloader
test_evaluator = val_evaluator
