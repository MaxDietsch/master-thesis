model = dict(
    type='SimSiam',
    backbone=dict(
        type='SwinTransformer',
        arch = 'small',
        img_size = 640),
    neck=dict(
        type='NonLinearNeck',
        in_channels=768,
        hid_channels=768,
        out_channels=768,
        num_layers=3,
        with_last_bn_affine=False,
        with_avg_pool=True),
    head=dict(
        type='LatentPredictHead',
        loss=dict(type='CosineSimilarityLoss'),
        predictor=dict(
            type='NonLinearNeck',
            in_channels=768,
            hid_channels=512,
            out_channels=768,
            with_avg_pool=False,
            with_last_bn=False,
            with_last_bias=True)),
)
