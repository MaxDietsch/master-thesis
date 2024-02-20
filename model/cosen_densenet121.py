model = dict(
    backbone=dict(arch='121', type='DenseNet'),
    head=dict(
        in_channels=1024,
        loss=dict(type='CoSenCrossEntropyLoss'),
        num_classes=4,
        topk=1,
        type='CoSenLinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='CoSenClassifier')
