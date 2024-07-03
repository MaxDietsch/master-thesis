

dataset_type = 'CustomDataset'
# TODO: change this 
data_preprocessor = dict(
        type='SelfSupDataPreprocessor',
        # Input image data channels in 'RGB' order
        mean=[130.36, 84.27, 72.21],
        std=[80.45, 62.92, 57.33],
        to_rgb=True,
        )


view_pipeline = [
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(type='RandomFlip', prob=0.5),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type = 'Resize', scale = (128, 128), interpolation = 'bicubic'),
    dict(type='MultiView', num_views=2, transforms=[view_pipeline]),
    dict(type='PackInputs', meta_keys=['img_path'])
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    drop_last=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root='../../SSL-HK',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=train_pipeline))


# is this needed ? actually not
"""
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    drop_last=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root='../../SELFSUP',
        ann_file='meta/val.txt',
        data_prefix=dict(img_path='val/'),
        pipeline=val_pipeline))
val_evaluator = ?? 
"""
