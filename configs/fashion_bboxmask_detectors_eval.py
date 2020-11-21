_base_ = [
    './_base_/models/cascade_mask_rcnn_r50_fpn_fashion.py',
    # './_base_/schedules/schedule_1x.py',
    './_base_/default_runtime.py'
]


model = dict(
    backbone=dict(
        type='DetectoRS_ResNet',
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True),
    neck=dict(
        type='RFP',
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            pretrained='torchvision://resnet50',
            style='pytorch')))

work_dir = './work_dirs/fashion_detectors'

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

# optimizer
# optimizer = dict(type='SGD', lr=0.008, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[20, 23])
total_epochs = 24

# checkpoint_config = dict(interval=1)
# # yapf:disable
# log_config = dict(
#     interval=50,
#     hooks=[
#         dict(type='TextLoggerHook'),
#         # dict(type='TensorboardLoggerHook')
#     ])
# # yapf:enable
# dist_params = dict(backend='nccl')
# log_level = 'INFO'
# load_from = None
# resume_from = None
dataset_type = 'FashionDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# classes = ('top',
#            'blouse',
#            't-shirt',
#            'Knitted fabric',
#            'shirt',
#            'bra top',
#            'hood',
#            'blue jeans',
#            'pants',
#            'skirt',
#            'leggings',
#            'jogger pants',
#            'coat',
#            'jacket',
#            'jumper',
#            'padding jacket',
#            'best',
#            'kadigan',
#            'zip up',
#            'dress',
#            'jumpsuit')

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'fashion/train_total.json',
        img_prefix=data_root + 'fashion/train_images',
        # classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'fashion/val_split.json',
        img_prefix=data_root + 'fashion/train_images',
        # classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'fashion/val_split.json',
        img_prefix=data_root + 'fashion/train_images',
        # classes=classes,
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
