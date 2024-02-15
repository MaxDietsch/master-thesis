optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001))

param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[100, 200, 300], gamma=0.1)

#train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)

train_cfg = dict(by_epoch=2, max_epochs=400, val_interval=1)

val_cfg = dict()
test_cfg = dict()

# If you use a different total batch size, like 512 and enable auto learning rate scaling.
# We will scale up the learning rate to 2 times.
#auto_scale_lr = dict(base_batch_size=256)

