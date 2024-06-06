model = dict(
    type='MoCo',
    queue_len=512,
    feat_dim=128,
    momentum=0.001,
    backbone=dict(
        type='EfficientNet',
        arch = 'b4'),
    neck=dict(
        type='MoCoV2Neck',
        in_channels=1792,
        hid_channels=1792,
        out_channels=128,
        with_avg_pool=True),
    head=dict(
        type='ContrastiveHead',
        loss=dict(type='CrossEntropyLoss'),
        temperature=0.2))
