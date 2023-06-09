# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Optional

import torch

from mmdet.core.bbox.builder import BBOX_CODERS

X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = list(range(11))
YAW = 6

@BBOX_CODERS.register_module()
class SparseBox3DDecoder(object):
    def __init__(
        self,
        num_output: int = 300,
        score_threshold: Optional[float] = None,
    ):
        super(SparseBox3DDecoder, self).__init__()
        self.num_output = num_output
        self.score_threshold = score_threshold

    def decode(self, cls_scores: torch.Tensor, box_preds: torch.Tensor):
        cls_scores = cls_scores[-1].sigmoid()
        box_preds = box_preds[-1]
        bs, num_pred, num_cls = cls_scores.shape
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            self.num_output, dim=1
        )
        cls_ids = indices % num_cls
        if self.score_threshold is not None:
            mask = cls_scores >= self.score_threshold
        output = []
        for i in range(bs):
            category_ids = cls_ids[i]
            scores = cls_scores[i]
            box = box_preds[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                box = box[mask[i]]

            yaw = torch.atan2(box[:, SIN_YAW], box[:, COS_YAW])
            box = torch.cat(
                [
                    box[:, [X, Y, Z]],
                    box[:, [W, L, H]].exp(),
                    yaw[:, None],
                    box[:, VX:],
                ],
                dim=-1,
            )

            output.append(
                {
                    "boxes_3d": box.cpu(),
                    "scores_3d": scores.cpu(),
                    "labels_3d": category_ids.cpu(),
                }
            )
        return output
