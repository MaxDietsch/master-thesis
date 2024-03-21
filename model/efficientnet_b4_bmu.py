model = dict(
    type='ImageClassifier',
    backbone=dict(type='EfficientNet', arch='b4'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='BMUHead',
        num_classes=4,
        in_channels=1792,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1),
    ))
