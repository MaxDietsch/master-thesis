model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='LinearVisionTransformer',
        img_size = 640,
        arch = 's',
        patch_size = 10,
        frozen_stages = 0,
        ),
    #neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=4,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1),
    ))

