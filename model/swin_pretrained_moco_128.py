model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer',
        arch='small',
        img_size=128,
        pad_small_map = True,
        init_cfg = dict(type = 'Pretrained', checkpoint='../work_dirs/phase4/swin/ssl_moco_128/lr_decr/epoch_200.pth', prefix = 'backbone')
        ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=4,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight = 1.0),
        topk=(1)
        )
)
