model = dict(
    type='CoSenClassifier',
    backbone=dict(
        type='SwinTransformer',
        arch='small',
        img_size=640,
        init_cfg = dict(type = 'Pretrained', checkpoint='../work_dirs/phase2/swin/pretrain/epoch_100.pth', prefix = 'backbone')
        ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='CoSenLinearClsHead',
        num_classes=4,
        in_channels=768,
        loss=dict(type='CoSenCrossEntropyLoss', num_classes = 4, learning_rate = 0.1),
        topk=(1),
    ))
