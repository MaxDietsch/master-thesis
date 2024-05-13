model = dict(
    type='CoSenClassifier',
    backbone=dict(type='EfficientNet', arch='b4', init_cfg = dict(type = 'Pretrained', checkpoint='../work_dirs/phase2/efficientnet_b4/pretrain/epoch_100.pth', prefix = 'backbone')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='CoSenLinearClsHead',
        num_classes=4,
        in_channels=1792,
        loss=dict(type='CoSenCrossEntropyLoss', num_classes = 4, learning_rate = 0.1),
        topk=(1),
    ))
