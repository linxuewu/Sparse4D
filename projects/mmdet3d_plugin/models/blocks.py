# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from mmcv.cnn import Linear, Scale, bias_init_with_prob
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn.bricks.transformer import FFN
from mmcv.utils import build_from_cfg
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
)

__all__ = [
    "LinearFusionModule",
    "DepthReweightModule",
    "SparseBox3DKeyPointsGenerator",
    "SparseBox3DRefinementModule",
    "DeformableFeatureAggregation",
    "SparseBox3DEncoder",
]

X, Y, Z, W, L, H, SIN_Y, COS_Y, VX, VY, VZ = list(range(11))


@PLUGIN_LAYERS.register_module()
class LinearFusionModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        alpha=0.9,
        beta=10,
        enable_temporal_weight=True,
    ):
        super(LinearFusionModule, self).__init__()
        self.embed_dims = embed_dims
        self.alpha = alpha
        self.beta = beta
        self.enable_temporal_weight = enable_temporal_weight
        self.fusion_layer = nn.Linear(self.embed_dims * 2, self.embed_dims)

    def init_weight(self):
        xavier_init(self.fusion_layer, distribution='uniform', bias=0.0)

    def forward(self, feature_1, feature_2, time_interval=None):
        if self.enable_temporal_weight:
            temp_weight = self.alpha ** torch.abs(time_interval * self.beta)
            feature_2 = torch.transpose(
                feature_2.transpose(0, -1) * temp_weight, 0, -1
            )
        return self.fusion_layer(torch.cat([feature_1, feature_2], dim=-1))


@PLUGIN_LAYERS.register_module()
class DepthReweightModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        min_depth=1,
        max_depth=56,
        depth_interval=5,
        ffn_layers=2,
    ):
        super(DepthReweightModule, self).__init__()
        self.embed_dims = embed_dims
        self.min_depth = min_depth
        self.depth_interval = depth_interval
        self.depths = np.arange(min_depth, max_depth + 1e-5, depth_interval)
        self.max_depth = max(self.depths)

        layers = []
        for i in range(ffn_layers):
            layers.append(
                FFN(
                    embed_dims=embed_dims,
                    feedforward_channels=embed_dims,
                    num_fcs=2,
                    act_cfg=dict(type="ReLU", inplace=True),
                    dropout=0.0,
                    add_residual=True,
                )
            )
        layers.append(nn.Linear(embed_dims, len(self.depths)))
        self.depth_fc = nn.Sequential(*layers)

    def forward(self, features, points_3d, output_conf=False):
        reference_depths = torch.norm(
            points_3d[..., :2], dim=-1, p=2, keepdim=True
        )
        reference_depths = torch.clip(
            reference_depths,
            max=self.max_depth - 1e-5,
            min=self.min_depth + 1e-5,
        )
        weights = (
            1
            - torch.abs(reference_depths - points_3d.new_tensor(self.depths))
            / self.depth_interval
        )

        top2 = weights.topk(2, dim=-1)[0]
        weights = torch.where(
            weights >= top2[..., 1:2], weights, weights.new_tensor(0.0)
        )
        scale = torch.pow(top2[..., 0:1], 2) + torch.pow(top2[..., 1:2], 2)
        confidence = self.depth_fc(features).softmax(dim=-1)
        confidence = torch.sum(weights * confidence, dim=-1, keepdim=True)
        confidence = confidence / scale

        if output_conf:
            return confidence
        return features * confidence


@PLUGIN_LAYERS.register_module()
class SparseBox3DRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        output_dim: int = 11,
        num_cls: int = 10,
        normalize_yaw=False,
        with_cls_branch=True,
    ):
        super(SparseBox3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.output_dim = output_dim
        self.num_cls = num_cls
        self.normalize_yaw = normalize_yaw

        self.layers = nn.Sequential(
            Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.embed_dims),

            Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.embed_dims),

            Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim),
        )
        self.with_cls_branch = with_cls_branch
        if with_cls_branch:
            self.cls_layers = nn.Sequential(
                Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.LayerNorm(self.embed_dims),
                Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.LayerNorm(self.embed_dims),
                Linear(self.embed_dims, self.num_cls),
            )

    def init_weight(self):
        if self.with_cls_branch:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.cls_layers[-1].bias, bias_init)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        time_interval: torch.Tensor = 1.0,
    ):
        output = self.layers(instance_feature + anchor_embed)
        output[..., [X, Y, Z, W, L, H]] = (
            output[..., [X, Y, Z, W, L, H]] + anchor[..., [X, Y, Z, W, L, H]]
        )
        if self.normalize_yaw:
            output[..., [SIN_Y, COS_Y]] = torch.nn.functional.normalize(
                output[..., [SIN_Y, COS_Y]], dim=-1
            )
        if self.output_dim > 8:
            if not isinstance(time_interval, torch.Tensor):
                time_interval = instance_feature.new_tensor(time_interval)
            translation = torch.transpose(output[..., VX:], 0, -1)
            velocity = torch.transpose(translation / time_interval, 0, -1)
            output[..., VX:] = velocity + anchor[..., VX:]
        return output

    def cls_forward(self, instance_feature: torch.Tensor):
        assert self.with_cls_branch, "Without classification layers !!!"
        return self.cls_layers(instance_feature)


@PLUGIN_LAYERS.register_module()
class SparseBox3DKeyPointsGenerator(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_learnable_pts: int = 0,
        fix_scale: Optional[Tuple[Tuple]] = None,
    ):
        super(SparseBox3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_learnable_pts = num_learnable_pts
        if fix_scale is None:
            fix_scale = ((0.0, 0.0, 0.0),)
        self.fix_scale = np.array(fix_scale)
        self.num_pts = len(self.fix_scale) + num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = Linear(self.embed_dims, num_learnable_pts * 3)

    def init_weight(self):
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def forward(
        self,
        anchor,
        instance_feature=None,
        T_cur2temp_list=None,
        cur_timestamp=None,
        temp_timestamps=None,
    ):
        bs, num_anchor = anchor.shape[:2]
        fix_scale = anchor.new_tensor(self.fix_scale)
        scale = fix_scale[None, None].tile([bs, num_anchor, 1, 1])
        if self.num_learnable_pts > 0 and instance_feature is not None:
            learnable_scale = (
                self.learnable_fc(instance_feature)
                .reshape(bs, num_anchor, self.num_learnable_pts, 3)
                .sigmoid()
                - 0.5
            )
            scale = torch.cat([scale, learnable_scale], dim=-2)
        key_points = scale * anchor[..., None, [W, L, H]].exp()
        rotation_mat = anchor.new_zeros([bs, num_anchor, 3, 3])
        rotation_mat[:, :, 0, 0] = anchor[:, :, COS_Y]
        rotation_mat[:, :, 0, 1] = -anchor[:, :, SIN_Y]
        rotation_mat[:, :, 1, 0] = anchor[:, :, SIN_Y]
        rotation_mat[:, :, 1, 1] = anchor[:, :, COS_Y]
        rotation_mat[:, :, 2, 2] = 1
        key_points = torch.matmul(
            rotation_mat[:, :, None], key_points[..., None]
        ).squeeze(-1)
        key_points = key_points + anchor[..., None, [X, Y, Z]]

        if (
            cur_timestamp is None
            or temp_timestamps is None
            or T_cur2temp_list is None
            or len(temp_timestamps) == 0
        ):
            return key_points

        temp_key_points_list = []
        velocity = anchor[..., VX:]
        for i, t_time in enumerate(temp_timestamps):
            time_interval = cur_timestamp - t_time
            translation = (
                velocity
                * time_interval.to(dtype=velocity.dtype)[:, None, None]
            )
            temp_key_points = key_points - translation[:, :, None]
            T_cur2temp = T_cur2temp_list[i].to(dtype=key_points.dtype)
            temp_key_points = (
                T_cur2temp[:, None, None, :3]
                @ torch.cat(
                    [
                        temp_key_points,
                        torch.ones_like(temp_key_points[..., :1]),
                    ],
                    dim=-1,
                ).unsqueeze(-1)
            )
            temp_key_points = temp_key_points.squeeze(-1)
            temp_key_points_list.append(temp_key_points)
        return key_points, temp_key_points_list

    @staticmethod
    def anchor_projection(
        anchor,
        T_cur2temp_list,
        cur_timestamp=None,
        temp_timestamps=None,
    ):
        temp_anchors = []
        for i in range(len(temp_timestamps)):
            temp_anchor = anchor.clone()
            vel = anchor[..., VX:]
            vel_dim = vel.shape[-1]
            T_cur2temp = torch.unsqueeze(
                T_cur2temp_list[i].to(dtype=anchor.dtype), dim=1
            )

            center = temp_anchor[..., [X, Y, Z]]
            if cur_timestamp is not None and temp_timestamps is not None:
                translation = vel.transpose(0, -1) * (
                    cur_timestamp - temp_timestamps[i]
                ).to(dtype=vel.dtype)
                translation = translation.transpose(0, -1)
                center = center - translation
            temp_anchor[..., [X, Y, Z]] = (
                torch.matmul(
                    T_cur2temp[..., :3, :3], center[..., None]
                ).squeeze(dim=-1)
                + T_cur2temp[..., :3, 3]
            )

            temp_anchor[..., [COS_Y, SIN_Y]] = torch.matmul(
                T_cur2temp[..., :2, :2], temp_anchor[..., [COS_Y, SIN_Y], None]
            ).squeeze(-1)

            temp_anchor[..., VX:] = torch.matmul(
                T_cur2temp[..., :vel_dim, :vel_dim], vel[..., None]
            ).squeeze(-1)

            temp_anchors.append(temp_anchor)
        return temp_anchors


@ATTENTION.register_module()
class DeformableFeatureAggregation(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        dropout: float = 0.1,
        kps_generator: dict = None,
        temporal_fusion_module=None,
        use_temporal_anchor_embed=True,
    ):
        super(DeformableFeatureAggregation, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_temporal_anchor_embed = use_temporal_anchor_embed

        self.dropout = nn.Dropout(dropout)
        kps_generator["embed_dims"] = embed_dims
        self.kps_generator = build_from_cfg(kps_generator, PLUGIN_LAYERS)
        self.num_pts = self.kps_generator.num_pts
        self.weights_fc = Linear(
            embed_dims, num_groups * num_cams * num_levels * self.num_pts
        )
        if temporal_fusion_module is not None:
            if "embed_dims" not in temporal_fusion_module:
                temporal_fusion_module["embed_dims"] = embed_dims
            self.temp_module = build_from_cfg(
                temporal_fusion_module, PLUGIN_LAYERS
            )
        else:
            self.temp_module = None
        self.output_proj = Linear(embed_dims, embed_dims)

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict,
        feature_queue=None,
        meta_queue=None,
        depth_module=None,
        anchor_encoder=None,
        **kwargs: dict,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        if feature_queue is not None and len(feature_queue) > 0:
            T_cur2temp_list = []
            for meta in meta_queue:
                T_cur2temp_list.append(
                    instance_feature.new_tensor([
                        x['T_global_inv'] @ metas["img_metas"][i]['T_global']
                        for i,x in enumerate(meta["img_metas"])
                    ])
                )
            key_points, temp_key_points_list = self.kps_generator(
                anchor,
                instance_feature,
                T_cur2temp_list,
                metas["timestamp"],
                [meta["timestamp"] for meta in meta_queue],
            )
            temp_anchors = self.kps_generator.anchor_projection(
                anchor,
                T_cur2temp_list,
                metas["timestamp"],
                [meta["timestamp"] for meta in meta_queue],
            )
            temp_anchor_embeds = [
                anchor_encoder(x)
                if self.use_temporal_anchor_embed and anchor_encoder is not None
                else None
                for x in temp_anchors
            ]
            time_intervals = [
                (metas["timestamp"] - x["timestamp"]).to(
                    dtype=instance_feature.dtype
                )
                for x in [metas] + meta_queue
            ]
        else:
            key_points = self.kps_generator(anchor, instance_feature)
            temp_key_points_list = (
                feature_queue
            ) = meta_queue = temp_anchor_embeds = temp_anchors = []
            time_intervals = [instance_feature.new_tensor([0])]

        if self.temp_module is not None and len(feature_queue) == 0:
            features = instance_feature.new_zeros(
                [bs, num_anchor, self.num_pts, self.embed_dims]
            )
        else:
            features = None

        if not self.use_temporal_anchor_embed or anchor_encoder is None:
            weights = self._get_weights(instance_feature, anchor_embed)

        for (
            temp_feature_maps,
            temp_metas,
            temp_key_points,
            temp_anchor_embed,
            temp_anchor,
            time_interval,
        ) in zip(
            feature_queue[::-1] + [feature_maps],
            meta_queue[::-1] + [metas],
            temp_key_points_list[::-1] + [key_points],
            temp_anchor_embeds[::-1] + [anchor_embed],
            temp_anchors[::-1] + [anchor],
            time_intervals[::-1],
        ):
            if self.use_temporal_anchor_embed and anchor_encoder is not None:
                weights = self._get_weights(instance_feature, temp_anchor_embed)
            temp_features_next = self.feature_sampling(
                temp_feature_maps,
                temp_key_points,
                temp_metas["projection_mat"],
                temp_metas.get("image_wh"),
            )
            temp_features_next = self.multi_view_level_fusion(
                temp_features_next, weights
            )
            if depth_module is not None:
                temp_features_next = depth_module(
                    temp_features_next, temp_anchor[:, :, None]
                )

            if features is None:
                features = temp_features_next
            elif self.temp_module is not None:
                features = self.temp_module(
                    features, temp_features_next, time_interval
                )
            else:
                features = features + temp_features_next

        features = features.sum(dim=2)  # fuse multi-point features
        output = self.output_proj(features)
        output = self.dropout(output) + instance_feature
        return output

    def _get_weights(self, instance_feature, anchor_embed):
        bs, num_anchor = instance_feature.shape[:2]
        weights = (
            self.weights_fc(instance_feature + anchor_embed)
            .reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
                1,
            )
        )
        return weights

    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)
        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        )
        points_2d = points_2d / image_wh[:, :, None, None]
        points_2d = points_2d * 2 - 1
        points_2d = points_2d.flatten(end_dim=1)

        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(
                    fm.flatten(end_dim=1), points_2d
                )
            )
        features = torch.stack(features, dim=1)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims

        return features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ):
        bs, num_anchor = weights.shape[:2]
        features = weights * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )
        return features


@POSITIONAL_ENCODING.register_module()
class SparseBox3DEncoder(BaseModule):
    def __init__(self, embed_dims: int = 256, vel_dims: int = 3):
        super().__init__()
        self.embed_dims = embed_dims
        self.vel_dims = vel_dims
        def embedding_layer(input_dims):
            return  nn.Sequential(
                nn.Linear(input_dims, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.LayerNorm(self.embed_dims),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.LayerNorm(self.embed_dims),
            )
        self.pos_fc = embedding_layer(3)
        self.size_fc = embedding_layer(3)
        self.yaw_fc = embedding_layer(2)
        if vel_dims > 0:
            self.vel_fc = embedding_layer(self.vel_dims)
        self.output_fc = embedding_layer(self.embed_dims)

    def forward(self, box_3d: torch.Tensor):
        pos_feat = self.pos_fc(box_3d[..., [X, Y, Z]])
        size_feat = self.size_fc(box_3d[..., [W, L, H]])
        yaw_feat = self.yaw_fc(box_3d[..., [SIN_Y, COS_Y]])
        output = pos_feat + size_feat + yaw_feat
        if self.vel_dims > 0:
            vel_feat = self.vel_fc(box_3d[..., VX:VX+self.vel_dims])
            output = output + vel_feat
        output = self.output_fc(output)
        return output
