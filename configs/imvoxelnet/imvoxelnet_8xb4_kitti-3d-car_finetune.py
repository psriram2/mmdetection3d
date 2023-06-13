_base_ = "./imvoxelnet_8xb4_kitti-3d-car.py"

# model settings
# model = dict(
#     train_cfg=dict(code_weight=[
#         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2
    # ]))
# optimizer
# optim_wrapper = dict(optimizer=dict(lr=0.0001))

train_dataloader = dict(
    batch_size=4,
    num_workers=1,)
load_from = 'checkpoints/imvoxelnet_4x8_kitti-3d-car_20210830_003014-3d0ffdf4.pth'
# load_from = 'checkpoints/kitti360smog_finetune_epoch.pth'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=3, val_interval=1)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, by_epoch=True))

# param_scheduler = [
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=2,
#         by_epoch=True,
#         milestones=[8, 11],
#         gamma=0.1)
# ]

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(
#         _delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001),
#     paramwise_cfg=dict(
#         custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}),
#     clip_grad=dict(max_norm=35., norm_type=2))