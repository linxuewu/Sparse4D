_base_ = [
    './default_runtime.py'
]

class_names = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone'
]

num_classes = len(class_names)
embed_dims = 256
num_groups = 8
num_decoder = 6
model = dict(
    type='Sparse4D',
    use_grid_mask=True,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        frozen_stages=1,
        norm_eval=True,
        style='caffe',
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        stage_with_dcn=(False, False, True, True),
        norm_cfg=dict(type='BN2d', requires_grad=False),
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
    ),
    img_neck=dict(
        type='FPN',
        num_outs=4,
        start_level=1,
        out_channels=embed_dims,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048],
    ),
    head=dict(
        type="Sparse4DHead",
        cls_threshold_to_reg=0.05,
        num_decoder=num_decoder,
        instance_bank=dict(
            type="InstanceBank",
            num_anchor=900,
            embed_dims=embed_dims,
            anchor="nuscenes_kmeans900.npy",
            anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
        ),
        anchor_encoder=dict(
            type="SparseBox3DEncoder",
            embed_dims=embed_dims,
            vel_dims=3,
        ),
        graph_model=dict(
            type="MultiheadAttention",
            embed_dims=embed_dims,
            num_heads=num_groups,
            batch_first=True,
            dropout=0.1,
        ),
        norm_layer=dict(type='LN', normalized_shape=embed_dims),
        ffn=dict(
            type="FFN",
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 2,
            num_fcs=2,
            ffn_drop=0.1,
            act_cfg=dict(type='ReLU', inplace=True),
        ),
        deformable_model=dict(
            type="DeformableFeatureAggregation",
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=4,
            num_cams=6,
            proj_drop=0.1,
            kps_generator=dict(
                type="SparseBox3DKeyPointsGenerator",
                num_learnable_pts=6,
                fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ],
            ),
        ),
        refine_layer=dict(
            type="SparseBox3DRefinementModule",
            embed_dims=embed_dims,
            num_cls=num_classes,
        ),
        sampler=dict(
            type="SparseBox3DTarget",
            cls_weight=2.0,
            box_weight=0.25,
            reg_weights=[2.0] * 3 + [1.0] * 7,
            cls_wise_reg_weights={
                class_names.index("traffic_cone"): [
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0
                ],
            },
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_reg=dict(type='L1Loss', loss_weight=0.25),
        gt_cls_key="gt_labels_3d",
        gt_reg_key="gt_bboxes_3d",
        decoder=dict(type="SparseBox3DDecoder"),
        reg_weights=[2.0] * 3 + [1.0] * 7,
        kps_generator=dict(
            type="SparseBox3DKeyPointsGenerator",
            fix_scale=[
                [0, 0, 0],
                [0.45, 0, 0],
                [-0.45, 0, 0],
                [0, 0.45, 0],
                [0, -0.45, 0],
                [0, 0, 0.45],
                [0, 0, -0.45],
            ],
        ),
        depth_module=dict(
            type="DepthReweightModule",
            embed_dims=embed_dims,
        ),
    ),
)

dataset_type = 'NuScenes3DDetTrackDataset'
data_root = 'data/nuscenes/'
anno_root = 'data/nuscenes_cam/'
file_client_args = dict(backend='disk')

img_crop_range = [260, 900, 0, 1600]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False
)
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type="CustomCropMultiViewImage", crop_range=img_crop_range),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False
    ),
    dict(
        type='CircleObjectRangeFilter',
        class_dist_thred=[55] * len(class_names)
    ),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='NuScenesSparse4DAdaptor'),
    dict(
        type='Collect3D',
        keys=[
            'gt_bboxes_3d',
            'gt_labels_3d',
            'img',
            "timestamp",
            "projection_mat",
            "image_wh",
        ],
        meta_keys=["timestamp", "T_global", "T_global_inv"],
    )
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type="CustomCropMultiViewImage", crop_range=img_crop_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False
    ),
    dict(type='NuScenesSparse4DAdaptor'),
    dict(
        type='Collect3D',
        keys=[
            'img',
            "timestamp",
            "projection_mat",
            "image_wh",
        ],
        meta_keys=["timestamp", "T_global", "T_global_inv"],
    )
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False
)

data_basic_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    box_type_3d='LiDAR',
    version='v1.0-trainval',
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        **data_basic_config,
        ann_file=anno_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        test_mode=False,
    ),
    val=dict(
        **data_basic_config,
        ann_file=anno_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        test_mode=True,
    ),
    test=dict(
        **data_basic_config,
        ann_file=anno_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

vis_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False
    ),
    dict(
        type='Collect3D',
        keys=['img'],
        meta_keys=["timestamp", "lidar2img"],
    )
]

total_epochs = 24
evaluation = dict(interval=24, pipeline=vis_pipeline)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from='fcos3d.pth'
