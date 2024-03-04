#construct dataloader and evaluator
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    # Input image data channels in 'RGB' order
    mean=[151.14, 102.69, 97.74],
    std=[70.03, 55.91, 54.73],
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),     # read image
    dict(type='Resize', scale=(640, 640), interpolation='bicubic'),
    dict(type='PackInputs'),         # prepare images and labels
]

test_pipeline = [
    dict(type='LoadImageFromFile'),     # read image
    dict(type='Resize', scale=(640, 640), interpolation='bicubic'),
    dict(type='PackInputs'),                 # prepare images and labels
]

train_dataloader = None

val_dataloader = dict(
    batch_size=1,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='../../B_E_P_N',
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
        dict(type='SingleLabelMetric', items=['precision', 'recall', 'f1-score'], average=None)                        
        ]



val_dataloader = dict(
    batch_size=1,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='../../B_E_P_N',
        ann_file='meta/test.txt',
        data_prefix='test',
        with_label=True,
        classes=['normal', 'polyps', 'barretts', 'esophagitis'],
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)

test_evaluator = val_evaluator
