# User Configuration
# ==============================================================================
# Select Model Size: 's', 'm', 'l', 'x'
model_size = 'l' 

# Dataset Selection (True/False)
use_sdg = True
use_lvis = True 
use_coco = True

# ==============================================================================

_base_ = (f'../../third_party/mmyolo/configs/yolov8/'
          f'yolov8_{model_size}_mask-refine_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)

# Class counts (approximate)
ycb_classes_count = 77
lvis_classes_count = 1203
coco_classes_count = 80

# Calculate total classes based on selection
num_classes = 0
if use_sdg: num_classes += ycb_classes_count
if use_lvis: num_classes += lvis_classes_count
if use_coco: num_classes += coco_classes_count

if num_classes == 0:
    raise ValueError("At least one dataset must be enabled!")

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
# Global Batch Size = 32 * 4 GPUs = 128 (matches standard YOLOv8 base config 8xb16)
train_batch_size_per_gpu = 32
train_num_workers = 8
load_from = f'../../weights/yolo_world_v2_{model_size}_stage2.pth'
text_model_name = 'openai/clip-vit-base-patch32'
persistent_workers = True

# model settings
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_training_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='SimpleStemMultiModalRotatedBackbone',
        backbone_cfg=dict(
            type='YOLOv8CSPDarknet',
            arch='P5',
            last_stage_out_channels=_base_.last_stage_out_channels,
            norm_cfg=_base_.norm_cfg,
            act_cfg=_base_.act_cfg,
            deepen_factor=_base_.deepen_factor,
            widen_factor=_base_.widen_factor,
            init_cfg=_base_.init_cfg),
        text_model_cfg=dict(
            type='CLIPTextModel',
            model_name=text_model_name,
            frozen_modules=['all'])),
    neck=dict(type='YOLOWorldPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
              text_enhancerv2=dict(type='ImagePoolingAttentionModule',
                                   embed_channels=256,
                                   num_heads=8)),
    bbox_head=dict(type='YOLOWorldHead',
                   head_module=dict(type='YOLOWorldHeadModule',
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes)),
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

# Datasets Definitions
datasets_list = []

if use_lvis:
    lvis_train_dataset = dict(
        type='YOLOv5LVISV1Dataset',
        data_root='data/lvis',
        ann_file='annotations/lvis_v1_train.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        class_text_path='data/texts/lvis_v1_class_texts.json',
        pipeline=train_pipeline)
    datasets_list.append(lvis_train_dataset)

if use_coco:
    coco_train_dataset = dict(
        type='YOLOv5CocoDataset',
        data_root='data/coco',
        ann_file='annotations/instances_val2017.json', # Using Validation set for training to save space
        data_prefix=dict(img='val2017/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        class_text_path='data/texts/coco_class_texts.json',
        pipeline=train_pipeline)
    datasets_list.append(coco_train_dataset)

if use_sdg:
    sdg_train_dataset = dict(
        type='MultiModalDataset',
        dataset=dict(
            type='YOLOv5CocoDataset',
            data_root='data/sdg/dataset',
            ann_file='train.json',
            data_prefix=dict(img='images/'),
            filter_cfg=dict(filter_empty_gt=False, min_size=32)),
        class_text_path='data/texts/sdg_texts.json',
        pipeline=train_pipeline)
    datasets_list.append(sdg_train_dataset)


train_dataloader = dict(
    persistent_workers=persistent_workers,
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    collate_fn=dict(type='yolow_collate'),
    dataset=dict(
        type='ConcatDataset',
        datasets=datasets_list,
        ignore_keys=['classes', 'palette']
    )
)

test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts'))
]

# Validation set (Default to SDG if enabled, else LVIS or COCO)
if use_sdg:
    val_dataset = dict(
        type='MultiModalDataset',
        dataset=dict(
            type='YOLOv5CocoDataset',
            data_root='data/sdg/dataset',
            ann_file='val.json', # Use same for val if no split
            data_prefix=dict(img='images/'),
            filter_cfg=dict(filter_empty_gt=False, min_size=32)),
        class_text_path='data/texts/sdg_texts.json',
        pipeline=test_pipeline)
    
    test_dataset = dict(
        type='MultiModalDataset',
        dataset=dict(
            type='YOLOv5CocoDataset',
            data_root='data/sdg/dataset',
            ann_file='test.json',
            data_prefix=dict(img='images/'),
            filter_cfg=dict(filter_empty_gt=False, min_size=32)),
        class_text_path='data/texts/sdg_texts.json',
        pipeline=test_pipeline)

elif use_lvis:
     val_dataset = dict(
        type='YOLOv5LVISV1Dataset',
        data_root='data/lvis',
        ann_file='annotations/lvis_v1_val.json',
        data_prefix=dict(img='val2017/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        class_text_path='data/texts/lvis_v1_class_texts.json',
        pipeline=test_pipeline)
     test_dataset = val_dataset

elif use_coco:
     val_dataset = dict(
        type='YOLOv5CocoDataset',
        data_root='data/coco',
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        class_text_path='data/texts/coco_class_texts.json',
        pipeline=test_pipeline)
     test_dataset = val_dataset


val_dataloader = dict(dataset=val_dataset)
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
                     ann_file=val_dataset['dataset']['data_root'] + '/' + val_dataset['dataset']['ann_file'],
                     metric='bbox')
