model = dict(
    type='ImageClassifier',
    backbone=dict(type='EfficientNet', arch='b4', init_cfg = dict(type = 'Pretrained', checkpoint='../work_dirs/phase4/efficientnet_b4/ssl_moco/lr_decr/epoch_200.pth', prefix = 'backbone')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=4,
        in_channels=1792,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1),
    ))
