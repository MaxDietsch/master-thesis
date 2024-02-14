model = dict(
    type = 'DOSClassifier',
    backbone=dict(arch='121', type='DenseNet'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        in_channels=1024,
        loss=dict(type='DOSLoss'),
        num_classes=4,
        topk=1,
        type='DOSHead'),
    )
