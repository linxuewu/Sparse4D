"""
mAP: 0.4392
mATE: 0.5998
mASE: 0.2694
mAOE: 0.4709
mAVE: 0.2916
mAAE: 0.1800
NDS: 0.5384
Eval time: 234.9s

Per-class results:
Object Class	AP	ATE	ASE	AOE	AVE	AAE
car	0.651	0.396	0.144	0.053	0.216	0.183
truck	0.392	0.620	0.204	0.086	0.237	0.200
bus	0.384	0.780	0.210	0.051	0.536	0.236
trailer	0.157	1.015	0.252	0.527	0.274	0.082
construction_vehicle	0.112	0.964	0.484	1.140	0.115	0.362
pedestrian	0.521	0.568	0.292	0.556	0.328	0.151
motorcycle	0.454	0.534	0.259	0.681	0.380	0.213
bicycle	0.411	0.479	0.279	0.998	0.247	0.012
traffic_cone	0.690	0.276	0.294	nan	nan	nan
barrier	0.620	0.366	0.277	0.145	nan	nan
"""
# ================ base config ===================
num_gpus = 8
batch_size = 8
num_iters_per_epoch = int(28130 // (num_gpus * batch_size))
num_epochs = 100
checkpoint_epoch_interval = 20

checkpoint_config = dict(interval=num_iters_per_epoch*checkpoint_epoch_interval)
log_config = dict(
    interval=51,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type="TensorboardLoggerHook")
    ],
)
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = None
load_from = None
resume_from = None
workflow = [("train", 1)]
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
fp16 = dict(loss_scale=32.0)
input_shape = (704, 256)

# ================== model ========================
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

num_classes = len(class_names)
embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
use_deformable_func = True  # setup.py needs to be executed
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
drop_out = 0.1

model = dict(
    type="Sparse4D",
    use_grid_mask=True,
    use_deformable_func=use_deformable_func,
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style="pytorch",
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=True),
        pretrained="ckpt/resnet50-19c8e357.pth",
    ),
    img_neck=dict(
        type="FPN",
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048],
    ),
    depth_branch=dict(  # for auxiliary supervision only
        type="DenseDepthNet",
        embed_dims=embed_dims,
        num_depth_layers=num_depth_layers,
        loss_weight=0.2,
    ),
    head=dict(
        type="Sparse4DHead",
        cls_threshold_to_reg=0.05,
        instance_bank=dict(
            type="InstanceBank",
            num_anchor=900,
            embed_dims=embed_dims,
            anchor="nuscenes_kmeans900.npy",
            anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
            num_temp_instances=600,
            confidence_decay=0.6,
            feat_grad=False,
        ),
        anchor_encoder=dict(
            type="SparseBox3DEncoder",
            embed_dims=embed_dims,
            vel_dims=3,
        ),
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=[
            "deformable",
            "ffn",
            "norm",
            "refine",
        ] * num_single_frame_decoder + [
            "temp_gnn",
            "gnn",
            "norm",
            "deformable",
            "ffn",
            "norm",
            "refine",
        ] * (num_decoder - num_single_frame_decoder),
        temp_graph_model=dict(
            type="MultiheadAttention",
            embed_dims=embed_dims,
            num_heads=num_groups,
            batch_first=True,
            dropout=drop_out,
        ),
        graph_model=dict(
            type="MultiheadAttention",
            embed_dims=embed_dims,
            num_heads=num_groups,
            batch_first=True,
            dropout=drop_out,
        ),
        norm_layer=dict(type="LN", normalized_shape=embed_dims),
        ffn=dict(
            type="AsymmetricFFN",
            in_channels=embed_dims * 2,
            pre_norm=dict(type="LN"),
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            num_fcs=2,
            ffn_drop=drop_out,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        deformable_model=dict(
            type="DeformableFeatureAggregation",
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=6,
            attn_drop=0.15,
            use_deformable_func=use_deformable_func,
            use_camera_embed=True,
            residual_mode="cat",
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
            refine_yaw=True,
        ),
        sampler=dict(
            type="SparseBox3DTarget",
            cls_weight=2.0,
            box_weight=0.25,
            reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
            cls_wise_reg_weights={
                class_names.index("traffic_cone"): [
                    2.0,
                    2.0,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                ],
            },
        ),
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_reg=dict(type="L1Loss", loss_weight=0.25),
        gt_cls_key="gt_labels_3d",
        gt_reg_key="gt_bboxes_3d",
        decoder=dict(type="SparseBox3DDecoder"),
        reg_weights=[2.0] * 3 + [1.0] * 7,
    ),
)

# ================== data ========================
dataset_type = "NuScenes3DDetTrackDataset"
data_root = "data/nuscenes/"
anno_root = "data/nuscenes_cam/"
file_client_args = dict(backend="disk")

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(type="ResizeCropFlipImage"),
    dict(
        type="MultiScaleDepthMapGenerator",
        downsample=strides[:num_depth_layers],
    ),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
    ),
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[55] * len(class_names),
    ),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect3D",
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            "gt_depth",
            "focal",
            "gt_bboxes_3d",
            "gt_labels_3d",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp"],
    ),
]
test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="DefaultFormatBundle3D", class_names=class_names, with_label=False
    ),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect3D",
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp"],
    ),
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

data_basic_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    box_type_3d="LiDAR",
    version="v1.0-trainval",
)

data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,
    train=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_train.pkl",
        pipeline=train_pipeline,
        test_mode=False,
        rot_range=[-0.3925, 0.3925],
        data_aug_conf=data_aug_conf,
        with_seq_flag=True,
        sequences_split_num=2,
        keep_consistent_seq_aug=True,
        filter_empty_gt=False,
    ),
    val=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_val.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
    ),
    test=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_val.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
    ),
)

# ================== training ========================
optimizer = dict(
    type="AdamW", lr=6e-4, weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.5),
        }
    ),
)
optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
runner = dict(
    type='IterBasedRunner',
    max_iters=num_iters_per_epoch * num_epochs,
)

# ================== eval ========================
vis_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="DefaultFormatBundle3D", class_names=class_names, with_label=False
    ),
    dict(
        type="Collect3D",
        keys=["img"],
        meta_keys=["timestamp", "lidar2img"],
    ),
]
evaluation = dict(
    interval=num_iters_per_epoch*checkpoint_epoch_interval,
    pipeline=vis_pipeline
)
