# Copyright (c) Horizon Robotics. All rights reserved.
from inspect import signature

import torch

from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    build_backbone,
    build_head,
    build_neck,
)
from .grid_mask import GridMask

try:
    from ..ops import DeformableAggregationFunction as DAF
except:
    DAF = None

__all__ = ["Sparse4D"]


@DETECTORS.register_module()
class Sparse4D(BaseDetector):
    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
    ):
        super(Sparse4D, self).__init__(init_cfg=init_cfg)
        if pretrained is not None:
            backbone.pretrained = pretrained
        self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        self.head = build_head(head)
        self.use_grid_mask = use_grid_mask
        self.use_deformable_func = use_deformable_func and DAF is not None
        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )
        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, metas.get("focal"))
        else:
            depths = None
        if self.use_deformable_func:
            feature_maps = DAF.feature_maps_format(feature_maps)
        if return_depth:
            return feature_maps, depths
        return feature_maps

    @force_fp32(apply_to=("img",))
    def forward(self, **data):
        if self.training:
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)

    def forward_train(self, **data):
        img = data.pop("img")
        feature_maps, depths = self.extract_feat(img, True, data)

        if "data_queue" in data or "future_data_queue" in data:
            feature_queue = []
            meta_queue = []
            with torch.no_grad():
                for d in data.get("data_queue", []) + data.get(
                    "future_data_queue", []
                ):
                    img = d.pop("img")
                    feature_queue.append(self.extract_feat(img))
                    meta_queue.append(d)
        else:
            feature_queue = None
            meta_queue = None

        cls_scores, reg_preds = self.head(
            feature_maps, data, feature_queue, meta_queue
        )
        if self.use_deformable_func:
            feature_maps = DAF.feature_maps_format(feature_maps, inverse=True)
        output = self.head.loss(cls_scores, reg_preds, data, feature_maps)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        return output

    def forward_test(self, **data):
        if isinstance(data["img"], list):
            return self.aug_test(**data)
        else:
            return self.simple_test(**data)

    def simple_test(self, **data):
        img = data.pop("img")
        feature_maps = self.extract_feat(img)

        if "future_data_queue" in data:
            feature_queue = []
            meta_queue = []
            with torch.no_grad():
                for d in data["future_data_queue"]:
                    img = d.pop("img")
                    feature_queue.append(self.extract_feat(img))
                    meta_queue.append(d)
        else:
            feature_queue = None
            meta_queue = None

        cls_scores, reg_preds = self.head(
            feature_maps, data, feature_queue, meta_queue
        )
        results = self.head.post_process(cls_scores, reg_preds)
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(**data)
