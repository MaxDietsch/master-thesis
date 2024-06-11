model = dict(
    type='RelativeLoc',
    backbone=dict(type='EfficientNet', arch='b4'),
    neck=dict(
        type='RelativeLocNeck',
        in_channels=1792,
        out_channels=3584,
        with_avg_pool=True),
    head=dict(
        type='LinearHead',
        loss=dict(type='CrossEntropyLoss'),
        in_channels=3584,
        num_classes=8,
        init_cfg=[
            dict(type='Normal', std=0.005, layer='Linear'),
            dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
        ]))
