model = dict(
    type='CoSenClassifier',
    backbone=dict(type='EfficientNet', arch='b4'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='CoSenLinearClsHead',
        num_classes=4,
        in_channels=1792,
        loss=dict(type='CoSenCrossEntropyLoss', num_classes = 4, learning_rate = 0.1),
        topk=(1),
    ))
