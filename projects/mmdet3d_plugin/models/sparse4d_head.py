# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmcv.runner import BaseModule, force_fp32
from mmcv.utils import build_from_cfg
from mmdet.models import HEADS, build_loss
from mmdet.core import reduce_mean
from mmdet.core.bbox.builder import build_bbox_coder, build_sampler

from .blocks import DeformableFeatureAggregation as DFG

__all__ = ["Sparse4DHead"]


@HEADS.register_module()
class Sparse4DHead(BaseModule):
    def __init__(
        self,
        num_anchor: int,
        anchor_file: str,
        num_decoder: int,
        embed_dims: int,
        anchor_encoder: dict,
        graph_model: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        depth_module: dict = None,
        loss_cls: dict = None,
        loss_reg: dict = None,
        decoder: dict = None,
        sampler: dict = None,
        gt_cls_key: str = "category_ids",
        gt_reg_key: str = "boxes",
        reg_weights=None,
        operation_order: Optional[List[str]] = None,
        pre_norm=False,
        kps_generator=None,
        default_time_interval=0.5,
        max_queue_length=0,
        cls_threshold_to_reg=-1,
        init_cfg=None,
        **kwargs,
    ):
        super(Sparse4DHead, self).__init__(init_cfg)
        self.num_anchor = num_anchor
        self.num_decoder = num_decoder
        self.embed_dims = embed_dims
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.default_time_interval = default_time_interval
        self.max_queue_length = max_queue_length
        self.cls_threshold_to_reg = cls_threshold_to_reg
        if reg_weights is None:
            self.reg_weights = [1.0] * 10
        else:
            self.reg_weights = reg_weights
        if operation_order is None:
            operation_order = [
                "gnn",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ]
        else:
            assert set(operation_order) == set(
                [
                    "gnn",
                    "norm",
                    "deformable",
                    "ffn",
                    "refine",
                    "identity",
                    "add",
                ]
            )
        self.operation_order = operation_order * num_decoder
        if pre_norm:
            self.pre_norm = nn.LayerNorm(embed_dims)
        else:
            self.pre_norm = None
        self.op_config_map = {
            "gnn": [graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "deformable": [deformable_model, ATTENTION],
            "refine": [refine_layer, PLUGIN_LAYERS],
        }

        self.anchor_encoder = build_from_cfg(
            anchor_encoder, POSITIONAL_ENCODING
        )
        self.layers = nn.ModuleList(
            [
                build_from_cfg(*self.op_config_map[op])
                if op in self.op_config_map else None
                for op in self.operation_order
            ]
        )
        anchor = np.load(anchor_file)[:self.num_anchor]
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32), requires_grad=True
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Embedding(self.num_anchor, self.embed_dims)
        if depth_module is not None:
            self.depth_module = build_from_cfg(depth_module, PLUGIN_LAYERS)
        else:
            self.depth_module = None
        if sampler is not None:
            self.sampler = build_sampler(sampler)
        if decoder is not None:
            self.decoder = build_bbox_coder(decoder)
        if loss_cls is not None:
            self.loss_cls = build_loss(loss_cls)
        if loss_reg is not None:
            self.loss_reg = build_loss(loss_reg)
        if kps_generator is not None:
            self.kps_generator = build_from_cfg(kps_generator, PLUGIN_LAYERS)
        else:
            self.kps_generator = None
        self.feature_queue = [] if max_queue_length > 0 else None
        self.meta_queue = [] if max_queue_length > 0 else None

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
        feature_queue=None,
        meta_queue=None,
    ):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]
        instance_feature = torch.tile(
            self.instance_feature.weight[None], (batch_size, 1, 1)
        )
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))
        anchor_embed = self.anchor_encoder(anchor)
        if self.pre_norm is not None:
            instance_feature = self.pre_norm(instance_feature)

        if not self.training:
            feature_queue = self.feature_queue
            meta_queue = self.meta_queue

        prediction = []
        classification = []
        if meta_queue is not None and len(meta_queue) > 0:
            time_interval = metas["timestamp"] - meta_queue[0]["timestamp"]
            time_interval = time_interval.to(dtype=instance_feature.dtype)
            time_interval = torch.where(
                time_interval == 0,
                time_interval.new_tensor(self.default_time_interval),
                time_interval,
            )
        else:
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * len(instance_feature)
            )
        for i, op in enumerate(self.operation_order):
            if op == "gnn":
                instance_feature = self.layers[i](
                    instance_feature,
                    query_pos=anchor_embed,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "identity":
                identity = instance_feature
            elif op == "add":
                instance_feature = instance_feature + identity
            elif op == "deformable":
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                    feature_queue=feature_queue,
                    meta_queue=meta_queue,
                    depth_module=self.depth_module,
                    anchor_encoder=self.anchor_encoder,
                )
            elif op == "refine":
                anchor = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                )
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
                    if self.training:
                        cls = self.layers[i].cls_forward(instance_feature)
                    else:
                        cls = None
                else:
                    cls = self.layers[i].cls_forward(instance_feature)
                prediction.append(anchor)
                classification.append(cls)

        if not self.training and feature_queue is not None:
            self.feature_queue.insert(0, feature_maps)
            self.meta_queue.insert(0, metas)
            while len(self.feature_queue) > self.max_queue_length:
                self.feature_queue.pop()
                self.meta_queue.pop()

        return classification, prediction

    @force_fp32(apply_to=('cls_scores', 'reg_preds'))
    def loss(self, cls_scores, reg_preds, data, feature_maps=None):
        output = {}
        for decoder_idx, (cls, reg) in enumerate(zip(cls_scores, reg_preds)):
            reg = reg[..., : len(self.reg_weights)]
            cls_target, reg_target, reg_weights = self.sampler.sample(
                cls,
                reg,
                data[self.gt_cls_key],
                data[self.gt_reg_key],
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            
            num_pos = max(
                reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
            )
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, cls.max(dim=-1).values.sigmoid() > threshold
                )

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos)

            mask = mask.reshape(-1)
            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )
            reg_loss = self.loss_reg(
                reg, reg_target, weight=reg_weights, avg_factor=num_pos
            )

            output.update(
                {
                    f"loss_cls_{decoder_idx}": cls_loss,
                    f"loss_reg_{decoder_idx}": reg_loss,
                }
            )

        if (
            self.depth_module is not None
            and self.kps_generator is not None
            and feature_maps is not None
        ):
            reg_target = self.sampler.encode_reg_target(
                data[self.gt_reg_key], reg_preds[0].device
            )
            loss_depth = []
            for i in range(len(reg_target)):
                key_points = self.kps_generator(reg_target[i][None])
                features = (
                    DFG.feature_sampling(
                        [f[i : i + 1] for f in feature_maps],
                        key_points,
                        data["projection_mat"][i:i+1],
                        data["image_wh"][i:i+1],
                    )
                    .mean(2)
                    .mean(2)
                )
                depth_confidence = self.depth_module(
                    features, reg_target[i][None, :, None], output_conf=True
                )
                loss_depth.append(-torch.log(depth_confidence).sum())
            output["loss_depth"] = (
                sum(loss_depth) / num_pos / self.kps_generator.num_pts
            )
        return output

    @force_fp32(apply_to=('cls_scores', 'reg_preds'))
    def post_process(self, cls_scores, reg_preds):
        return self.decoder.decode(cls_scores, reg_preds)
