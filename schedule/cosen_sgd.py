optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[100, 200, 300], gamma=0.1)

""" EpochBasedTrainLoop
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
"""

""" DosTrainLoop
k = [0, 4, 2, 1]
r = [0, 2, 2, 1]
samples_per_class = [150, 45, 132, 539]
train_cfg = dict(by_epoch=2, k = k, r = r, samples_per_class = samples_per_class, max_epochs=400, val_interval=1)
"""

#""" CoSenTrainLoop 
s_freq = 3
s_samples_per_class = [10, 10, 10, 10]
samples_per_class = [3312, 45, 132, 539]
mu1 = 0.5
mu2 = 0.5
s1 = 0.5
s2 = 0.5
train_cfg = dict(by_epoch = 3, s_freq = s_freq, s_samples_per_class = s_samples_per_class, samples_per_class = samples_per_class,
                 mu1 = mu1, mu2 = mu2, s1 = s1, s2 = s2, max_epochs = 400, val_interval = 1)
#"""


val_cfg = dict()
test_cfg = dict()

# If you use a different total batch size, like 512 and enable auto learning rate scaling.
# We will scale up the learning rate to 2 times.
#auto_scale_lr = dict(base_batch_size=256)

