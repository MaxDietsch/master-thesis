model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer',
        arch='small',
        img_size=640,
        init_cfg = dict(type = 'Pretrained', checkpoint='../work_dirs/phase2/swin/pretrain/epoch_100.pth', prefix = 'backbone')
        ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight = 1.0),
        topk=(1)
        )
)
