model = dict(
    type='SimSiam',
    backbone=dict(
        type='EfficientNet',
        arch = 'b4'),
    neck=dict(
        type='NonLinearNeck',
        in_channels=1792,
        hid_channels=1792,
        out_channels=1792,
        num_layers=3,
        with_last_bn_affine=False,
        with_avg_pool=True),
    head=dict(
        type='LatentPredictHead',
        loss=dict(type='CosineSimilarityLoss'),
        predictor=dict(
            type='NonLinearNeck',
            in_channels=1792,
            hid_channels=512,
            out_channels=1792,
            with_avg_pool=False,
            with_last_bn=False,
            with_last_bias=True)),
)
