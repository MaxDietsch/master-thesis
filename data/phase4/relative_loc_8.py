dataset_type = 'CustomDataset'
data_root = '../../SSL-HK'

data_preprocessor=dict(
        type='RelativeLocDataPreprocessor',
        mean=[130.36, 84.27, 72.21],
        std=[80.45, 62.92, 57.33],
        to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomGrayscale', prob=0.66, keep_channels=True),
    dict(type='RandomPatchWithLabels'),
    dict(
        type='PackInputs',
        pseudo_label_keys=['patch_box', 'patch_label', 'unpatched_img'],
        meta_keys=['img_path'])
]




train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=train_pipeline))
