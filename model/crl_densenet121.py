model = dict(
    backbone=dict(arch='121', type='DenseNet'),
    head=dict(
        in_channels=1024,
        loss=dict(loss_weight=1.0, type='CRLLoss', min_classes = [1, 2, 3], k = 3, use_soft = False),
        num_classes=4,
        topk=1,
        type='LinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')
