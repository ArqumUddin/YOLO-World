# User Configuration
# ==============================================================================
# Select Model Size: 's', 'm', 'l', 'x'
model_size = 'l' 

# ==============================================================================

_base_ = ('../../third_party/mmyolo/configs/yolov8/'
          'yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)

# hyper-parameters
num_classes = 55  # YCB classes count (actual)
num_training_classes = num_classes
max_epochs = 80
close_mosaic_epochs = 10
save_epoch_intervals = 5
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4
weight_decay = 0.05
# Optimized for 4x NVIDIA L40S (48GB VRAM)
# 4x L40S has ~45GB usable per GPU. 
# Batch 48 caused OOM. Reducing to 32.
train_batch_size_per_gpu = 24
train_num_workers = 8
load_from = 'weights/yolo_world_v2_l_stage2.pth'
resume = True
text_model_name = 'openai/clip-vit-base-patch32'
persistent_workers = True

# model settings
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_training_classes,
    data_preprocessor=dict(type='mmdet.DetDataPreprocessor',
                           mean=[0.0, 0.0, 0.0],
                           std=[255.0, 255.0, 255.0],
                           bgr_to_rgb=True),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model=dict(
            type='YOLOv8CSPDarknet',
            arch='P5',
            last_stage_out_channels=_base_.last_stage_out_channels,
            norm_cfg=_base_.norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True),
            deepen_factor=_base_.deepen_factor,
            widen_factor=_base_.widen_factor),
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=text_model_name,
            frozen_modules=['all'])),
    neck=dict(type='YOLOWorldPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
    bbox_head=dict(type='YOLOWorldHead',
                   head_module=dict(type='YOLOWorldHeadModule',
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes,
                                    reg_max=16),
                   loss_dfl=dict(
                       type='mmdet.DistributionFocalLoss',
                       reduction='mean',
                       loss_weight=1.5 / 4)),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

# dataset settings
train_pipeline = [
    *_base_.pre_transform,
    dict(type='Mosaic',
         img_scale=_base_.img_scale,
         pad_val=114.0,
         pre_transform=_base_.pre_transform),
    dict(type='YOLOv5RandomAffine',
         max_rotate_degree=0.0,
         max_shear_degree=0.0,
         scaling_ratio_range=(0.5, 1.5),
         border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
         border_val=(114, 114, 114)),
    *_base_.last_transform[:-1],
    dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]

text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]

train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='data/sdg/dataset',
        ann_file='train.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/sdg_texts.json',
    pipeline=train_pipeline)

train_dataloader = dict(
    persistent_workers=persistent_workers,
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    collate_fn=dict(type='pseudo_collate'), # Override base collate with default pseudo_collate
    dataset=train_dataset
)

test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts'))
]

val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='data/sdg/dataset',
        ann_file='val.json', # Use same for val if no split
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/sdg_texts.json',
    pipeline=test_pipeline)

val_dataloader = dict(dataset=val_dataset)

test_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='data/sdg/dataset',
        ann_file='test.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/sdg_texts.json',
    pipeline=test_pipeline)

test_dataloader = dict(dataset=test_dataset)

default_hooks = dict(param_scheduler=dict(scheduler_type='linear',
                                          lr_factor=0.01,
                                          max_epochs=max_epochs),
                     checkpoint=dict(max_keep_ckpts=-1,
                                     save_best=None,
                                     interval=save_epoch_intervals))

custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=_base_.train_pipeline_stage2)
]

train_cfg = dict(max_epochs=max_epochs,
                 val_interval=5,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])

optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu),
    paramwise_cfg=dict(
        custom_keys={'backbone.text_model': dict(lr_mult=0.01),
                     'logit_scale': dict(weight_decay=0.0)}),
    constructor='YOLOWv5OptimizerConstructor')

val_evaluator = dict(_delete_=True,
                     type='mmdet.CocoMetric',
                     proposal_nums=(100, 1, 10),
                     ann_file='data/sdg/dataset/val.json',
                     metric='bbox')

visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend')])
