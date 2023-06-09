import torch
import random
import math
import os
from os import path as osp
import cv2
import tempfile
import copy

import numpy as np
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.eval.detection.config import config_factory as det_configs
from nuscenes.eval.common.config import config_factory as track_configs

import mmcv
from mmcv.utils import print_log
from mmdet.datasets import DATASETS
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.datasets.pipelines import Compose


@DATASETS.register_module()
class NuScenes3DDetTrackDataset(Custom3DDataset):
    NameMapping = {
        "movable_object.barrier": "barrier",
        "vehicle.bicycle": "bicycle",
        "vehicle.bus.bendy": "bus",
        "vehicle.bus.rigid": "bus",
        "vehicle.car": "car",
        "vehicle.construction": "construction_vehicle",
        "vehicle.motorcycle": "motorcycle",
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "movable_object.trafficcone": "traffic_cone",
        "vehicle.trailer": "trailer",
        "vehicle.truck": "truck",
    }
    DefaultAttribute = {
        "car": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.moving",
        "motorcycle": "cycle.without_rider",
        "construction_vehicle": "vehicle.parked",
        "bicycle": "cycle.without_rider",
        "barrier": "",
        "traffic_cone": "",
    }
    AttrMapping = {
        "cycle.with_rider": 0,
        "cycle.without_rider": 1,
        "pedestrian.moving": 2,
        "pedestrian.standing": 3,
        "pedestrian.sitting_lying_down": 4,
        "vehicle.moving": 5,
        "vehicle.parked": 6,
        "vehicle.stopped": 7,
    }
    AttrMapping_rev = [
        "cycle.with_rider",
        "cycle.without_rider",
        "pedestrian.moving",
        "pedestrian.standing",
        "pedestrian.sitting_lying_down",
        "vehicle.moving",
        "vehicle.parked",
        "vehicle.stopped",
    ]
    ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }
    CLASSES = (
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
    )
    ID_COLOR_MAP = [
        (59, 59, 238),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 255),
        (0, 127, 255),
        (71, 130, 255),
        (127, 127, 0),
    ]

    def __init__(
        self,
        ann_file,
        pipeline=None,
        data_root=None,
        classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        det3d_eval_version="detection_cvpr_2019",
        track3d_eval_version="tracking_nips_2019",
        version="v1.0-trainval",
        use_valid_flag=False,
        vis_score_threshold=0.35,
        seq_frame=0,
        max_tracking_frame_interval=1,
        max_interval=1,
        min_interval=1,
        max_time_interval=5,
        fix_interval=True,
        future_frame=0,
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[1.0, 1.0],
        translation_std=[0, 0, 0],
        data_aug_conf=None,
        tracking=False,
        sequences_split_num=1,
        with_seq_flag=False,
        keep_consistent_seq_aug=True,
    ):
        self.version = version
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )

        self.with_velocity = with_velocity
        self.det3d_eval_version = det3d_eval_version
        self.det3d_eval_configs = det_configs(self.det3d_eval_version)
        self.track3d_eval_version = track3d_eval_version
        self.track3d_eval_configs = track_configs(self.track3d_eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        self.vis_score_threshold = vis_score_threshold

        self.seq_frame = seq_frame
        self.max_tracking_frame_interval = max_tracking_frame_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.max_time_interval = max_time_interval
        self.fix_interval = fix_interval
        self.future_frame = future_frame

        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std
        self.data_aug_conf = data_aug_conf
        self.tracking = tracking
        self.sequences_split_num = sequences_split_num
        self.keep_consistent_seq_aug = keep_consistent_seq_aug
        self.current_aug = None
        self.last_id = None
        if with_seq_flag:
            self._set_sequence_group_flag()

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and len(self.data_infos[idx]["sweeps"]) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.sequences_split_num != 1:
            if self.sequences_split_num == "all":
                self.flag = np.array(
                    range(len(self.data_infos)), dtype=np.int64
                )
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(
                            range(
                                0,
                                bin_counts[curr_flag],
                                math.ceil(
                                    bin_counts[curr_flag]
                                    / self.sequences_split_num
                                ),
                            )
                        )
                        + [bin_counts[curr_flag]]
                    )

                    for sub_seq_idx in (
                        curr_sequence_length[1:] - curr_sequence_length[:-1]
                    ):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert (
                    len(np.bincount(new_flags))
                    == len(np.bincount(self.flag)) * self.sequences_split_num
                )
                self.flag = np.array(new_flags, dtype=np.int64)

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int(
                    (1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"]))
                    * newH
                )
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def pre_pipeline(
        self, results, rot_angle=None, scale_ratio=None, aug_configs=None
    ):
        super().pre_pipeline(results)
        if rot_angle is None:
            results["rot_angle"] = np.random.uniform(*self.rot_range)
        else:
            results["rot_angle"] = rot_angle
        if scale_ratio is None:
            results["scale_ratio"] = np.random.uniform(*self.scale_ratio_range)
        else:
            results["scale_ratio"] = scale_ratio
        if self.data_aug_conf is not None and aug_configs is None:
            results["aug_configs"] = self._sample_augmentation()
        else:
            results["aug_configs"] = aug_configs
        if not self.test_mode and "instance_inds" in results["ann_info"]:
            results["instance_inds"] = results["ann_info"]["instance_inds"]
        results["seq_frame"] = self.seq_frame
        results["tracking"] = self.tracking
        results["test_mode"] = self.test_mode
        return (
            results["rot_angle"],
            results["scale_ratio"],
            results["aug_configs"],
        )

    def _getitem(
        self, idx, rot_angle=None, scale_ratio=None, aug_configs=None
    ):
        if self.test_mode:
            data, aug_configs = self.prepare_test_data(idx, aug_configs)
            return data, None, None, aug_configs
        while True:
            (
                data,
                rot_angle,
                scale_ratio,
                aug_configs,
            ) = self.prepare_train_data(
                idx,
                rot_angle=rot_angle,
                scale_ratio=scale_ratio,
                aug_configs=aug_configs,
            )
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data, rot_angle, scale_ratio, aug_configs

    def prepare_test_data(self, index, aug_configs=None):
        input_dict = self.get_data_info(index)
        aug_configs = self.pre_pipeline(input_dict, aug_configs=aug_configs)[
            -1
        ]
        example = self.pipeline(input_dict)
        return example, aug_configs

    def prepare_train_data(
        self, index, rot_angle=None, scale_ratio=None, aug_configs=None
    ):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None, rot_angle, scale_ratio, aug_configs
        rot_angle, scale_ratio, aug_configs = self.pre_pipeline(
            input_dict,
            rot_angle=rot_angle,
            scale_ratio=scale_ratio,
            aug_configs=aug_configs,
        )
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and (
            example is None or ~(example["gt_labels_3d"]._data != -1).any()
        ):
            return None, rot_angle, scale_ratio, aug_configs
        return example, rot_angle, scale_ratio, aug_configs

    def __getitem__(self, idx):
        if isinstance(idx, dict):
            rot_angle, scale_ratio, aug_configs = idx["aug"]
            idx = idx["idx"]
            data = self._getitem(idx, rot_angle, scale_ratio, aug_configs)[0]
        else:
            data, rot_angle, scale_ratio, aug_configs = self._getitem(idx)
        # if (
        #     not self.keep_consistent_seq_aug
        #     or idx == 0
        #     or self.flag[idx] != self.flag[idx - 1]
        #     or self.current_aug is None
        # ):
        #     data, rot_angle, scale_ratio, aug_configs = self._getitem(idx)
        #     self.current_aug = rot_angle, scale_ratio, aug_configs
        # else:
        #     data = self._getitem(idx, *self.current_aug)[0]
        #     rot_angle, scale_ratio, aug_configs = self.current_aug

        if isinstance(data["img_metas"], list):
            cur_timestamp = data["img_metas"][0].data["timestamp"]
        else:
            cur_timestamp = data["img_metas"].data["timestamp"]

        interval = (
            int(random.random() * self.max_interval) + self.min_interval
        )
        if self.seq_frame > 0 and not self.test_mode:
            seq_frame_indice = []
            seq_frame_num = self.seq_frame
            if self.tracking:
                seq_frame_num += (
                    int(
                        random.random()
                        * (self.max_tracking_frame_interval + 1)
                    )
                    + 1
                )

            idx_next = idx
            for i in range(seq_frame_num):
                if idx_next == 0:
                    break
                idx_next -= interval
                if not self.fix_interval:
                    interval = (
                        int(random.random() * self.max_interval)
                        + self.min_interval
                    )
                if (
                    i >= self.seq_frame
                    and seq_frame_num - i > self.seq_frame + 1
                ):
                    continue
                idx_next = max(idx_next, 0)
                seq_frame_indice.append(idx_next)

            data_queue = []
            for seq_idx in seq_frame_indice:
                seq_data = self._getitem(
                    seq_idx, rot_angle, scale_ratio, aug_configs
                )[0]
                if isinstance(seq_data["img_metas"], list):
                    seq_timestamp = seq_data["img_metas"][0].data["timestamp"]
                else:
                    seq_timestamp = seq_data["img_metas"].data["timestamp"]
                if abs(seq_timestamp - cur_timestamp) > self.max_time_interval:
                    break
                data_queue.append(seq_data)
            data["data_queue"] = data_queue

            if len(data_queue) > 0 and "instance_inds" in data:
                last_frame_id = max(len(data_queue) - self.seq_frame - 1, 0)
                last_instance_inds = data["data_queue"][last_frame_id][
                    "instance_inds"
                ]
                match_flag = (
                    last_instance_inds[None] == data["instance_inds"][:, None]
                )
                dummy_flag = np.logical_not(
                    match_flag.any(axis=-1, keepdims=True)
                )
                match_flag = np.concatenate([dummy_flag, match_flag], axis=-1)
                match_inds = np.where(match_flag)[1] - 1
                data["match_inds"] = match_inds

        if self.future_frame > 0:
            idx_next = idx
            future_data_queue = []
            if self.test_mode:
                interval = 1
            for i in range(self.future_frame):
                if idx_next == self.__len__() - 1:
                    break
                idx_next += interval
                if not self.fix_interval:
                    interval = (
                        int(random.random() * self.max_interval)
                        + self.min_interval
                    )
                idx_next = min(idx_next, self.__len__() - 1)
                future_data = self._getitem(
                    idx_next, rot_angle, scale_ratio, aug_configs
                )[0]
                if isinstance(future_data["img_metas"], list):
                    seq_timestamp = future_data["img_metas"][0].data[
                        "timestamp"
                    ]
                else:
                    seq_timestamp = future_data["img_metas"].data["timestamp"]
                if abs(seq_timestamp - cur_timestamp) > self.max_time_interval:
                    break
                future_data_queue.append(future_data)
            data["future_data_queue"] = future_data_queue
        return data

    def get_cat_ids(self, idx):
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        print(self.metadata)
        return data_infos

    def get_data_info(self, index):
        info = self.data_infos[index]
        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"] / 1e6,
            lidar2ego_translation=info["lidar2ego_translation"],
            lidar2ego_rotation=info["lidar2ego_rotation"],
            ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"],
        )
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = pyquaternion.Quaternion(
            info["lidar2ego_rotation"]
        ).rotation_matrix
        lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])
        ego2global = np.eye(4)
        ego2global[:3, :3] = pyquaternion.Quaternion(
            info["ego2global_rotation"]
        ).rotation_matrix
        ego2global[:3, 3] = np.array(info["ego2global_translation"])
        input_dict["lidar2global"] = ego2global @ lidar2ego

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            cam_intrinsic = []
            for cam_type, cam_info in info["cams"].items():
                image_paths.append(cam_info["data_path"])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
                lidar2cam_t = (
                    cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
                )
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info["cam_intrinsic"]
                cam_intrinsic.append(intrinsic)
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt.T
                lidar2img_rts.append(lidar2img_rt)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsic,
                )
            )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict["ann_info"] = annos

        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5),
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )
        if "instance_inds" in info:
            instance_inds = np.array(info["instance_inds"], dtype=np.int)[mask]
            anns_results["instance_inds"] = instance_inds
        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None):
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det)
            sample_token = self.data_infos[sample_id]["token"]
            boxes, sampled_points = lidar_nusc_box_to_global(
                self.data_infos[sample_id],
                boxes,
                mapped_class_names,
                self.det3d_eval_configs,
                self.det3d_eval_version,
                det.get("sampled_points", None),
            )
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if self.tracking and name in [
                    "barrier",
                    "traffic_cone",
                    "construction_vehicle",
                ]:
                    continue
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = NuScenes3DDetTrackDataset.DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = NuScenes3DDetTrackDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                )
                if sampled_points[i] is not None:
                    nusc_anno.update(sampled_points=sampled_points[i].tolist())
                if not self.tracking:
                    nusc_anno.update(
                        dict(
                            detection_name=name,
                            detection_score=box.score,
                            attribute_name=attr,
                        )
                    )
                else:
                    nusc_anno.update(
                        dict(
                            tracking_name=name,
                            tracking_score=box.score,
                            tracking_id=str(box.token),
                        )
                    )

                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_nusc.json")
        print("Results writes to", res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(
        self, result_path, logger=None, result_name="img_bbox"
    ):

        from nuscenes import NuScenes

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False
        )
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
        }
        if not self.tracking:
            from nuscenes.eval.detection.evaluate import NuScenesEval

            nusc_eval = NuScenesEval(
                nusc,
                config=self.det3d_eval_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
            )
            nusc_eval.main(render_curves=False)

            # record metrics
            metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
            detail = dict()
            metric_prefix = f"{result_name}_NuScenes"
            for name in self.CLASSES:
                for k, v in metrics["label_aps"][name].items():
                    val = float("{:.4f}".format(v))
                    detail[
                        "{}/{}_AP_dist_{}".format(metric_prefix, name, k)
                    ] = val
                for k, v in metrics["label_tp_errors"][name].items():
                    val = float("{:.4f}".format(v))
                    detail["{}/{}_{}".format(metric_prefix, name, k)] = val
                for k, v in metrics["tp_errors"].items():
                    val = float("{:.4f}".format(v))
                    detail[
                        "{}/{}".format(metric_prefix, self.ErrNameMapping[k])
                    ] = val

            detail["{}/NDS".format(metric_prefix)] = metrics["nd_score"]
            detail["{}/mAP".format(metric_prefix)] = metrics["mean_ap"]
        else:
            from nuscenes.eval.tracking.evaluate import TrackingEval

            nusc_eval = TrackingEval(
                config=self.track3d_eval_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
                nusc_version=self.version,
                nusc_dataroot=self.data_root,
            )
            metrics = nusc_eval.main()

            # record metrics
            metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
            print(metrics)
            detail = dict()
            metric_prefix = f"{result_name}_NuScenes"
            keys = [
                "amota",
                "amotp",
                "recall",
                "motar",
                "gt",
                "mota",
                "motp",
                "mt",
                "ml",
                "faf",
                "tp",
                "fp",
                "fn",
                "ids",
                "frag",
                "tid",
                "lgd",
            ]
            for key in keys:
                detail["{}/{}".format(metric_prefix, key)] = metrics[key]

        return detail

    def format_results(self, results, jsonfile_prefix=None, metric="bbox"):
        assert isinstance(results, list), "results must be a list"

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        if not ("pts_bbox" in results[0] or "img_bbox" in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            result_files = dict()
            for name in results[0]:
                print(f"\nFormating bboxes of {name}")
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)}
                )
        return result_files, tmp_dir

    def evaluate(
        self,
        results,
        metric=None,
        logger=None,
        jsonfile_prefix=None,
        result_names=["img_bbox"],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                ret_dict = self._evaluate_single(result_files[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files)
        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show or out_dir:
            self.show(results, save_dir=out_dir, show=show, pipeline=pipeline)
        return results_dict

    def _build_default_pipeline(self):
        pipeline = [
            dict(
                type="LoadPointsFromFile",
                coord_type="LIDAR",
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend="disk"),
            ),
            dict(
                type="LoadPointsFromMultiSweeps",
                sweeps_num=10,
                file_client_args=dict(backend="disk"),
            ),
            dict(
                type="DefaultFormatBundle3D",
                class_names=self.CLASSES,
                with_label=False,
            ),
            dict(type="Collect3D", keys=["points"]),
        ]
        return Compose(pipeline)

    def show(self, results, save_dir=None, show=False, pipeline=None):
        save_dir = "./" if save_dir is None else save_dir
        save_dir = os.path.join(save_dir, "visual")
        print_log(os.path.abspath(save_dir))
        pipeline = self._get_pipeline(pipeline)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        videoWriter = None

        for i, result in enumerate(results):
            if "img_bbox" in result.keys():
                result = result["img_bbox"]
            # data_info = self.data_infos.__getitem__(i)
            data_info = pipeline(self.get_data_info(i))
            imgs = []
            raw_imgs = [
                x.permute(1, 2, 0).cpu().numpy() for x in data_info["img"].data
            ]
            lidar2img = data_info["img_metas"].data["lidar2img"]
            # gt_bboxes_3d = self.get_ann_info(i)['gt_bboxes_3d']
            pred_bboxes_3d = result["boxes_3d"][
                result["scores_3d"] > self.vis_score_threshold
            ]
            if not isinstance(pred_bboxes_3d, LiDARInstance3DBoxes):
                pred_bboxes_3d = LiDARInstance3DBoxes(
                    pred_bboxes_3d,
                    box_dim=pred_bboxes_3d.shape[-1],
                    origin=(0.5, 0.5, 0.5),
                ).convert_to(self.box_mode_3d)

            if "sampled_points" in result and "track_ids" not in result:
                sampled_points = result["sampled_points"][
                    result["scores_3d"] > self.vis_score_threshold
                ]
            else:
                sampled_points = None

            if "track_ids" in result:
                color = []
                for id in result["track_ids"].cpu().numpy().tolist():
                    color.append(
                        self.ID_COLOR_MAP[int(id % len(self.ID_COLOR_MAP))]
                    )
            elif "labels_3d" in result:
                color = []
                for id in result["labels_3d"].cpu().numpy().tolist():
                    color.append(self.ID_COLOR_MAP[id])
            else:
                color = (255, 0, 0)

            for j, img_origin in enumerate(raw_imgs):
                img = img_origin.copy()
                if len(pred_bboxes_3d) != 0:
                    img = draw_lidar_bbox3d_on_img(
                        pred_bboxes_3d,
                        img,
                        lidar2img[j],
                        img_metas=None,
                        color=color,
                        thickness=3,
                    )
                    if sampled_points is not None:
                        img = draw_points_on_img(
                            sampled_points,
                            img,
                            lidar2img[j],
                            color=color,
                            circle=4,
                        )

                imgs.append(img)

            bev_range = 124  # meters
            marking_color = (127, 127, 127)
            bev_h, bev_w = img.shape[0] * 2, img.shape[0] * 2
            # bev_h, bev_w = img.shape[1], img.shape[1]
            bev = np.zeros([bev_h, bev_w, 3])
            bev_resolution = 124 / bev_h
            for cir in range(int(bev_range / 2 / 10)):
                cv2.circle(
                    bev,
                    (int(bev_h / 2), int(bev_w / 2)),
                    int((cir + 1) * 10 / bev_resolution),
                    marking_color,
                    thickness=3,
                )
            cv2.line(
                bev,
                (0, int(bev_h / 2)),
                (bev_w, int(bev_h / 2)),
                marking_color,
            )
            cv2.line(
                bev,
                (int(bev_w / 2), 0),
                (int(bev_w / 2), bev_h),
                marking_color,
            )
            if len(pred_bboxes_3d) != 0:
                bev_corners = pred_bboxes_3d.corners[:, [0, 3, 4, 7]][
                    ..., [0, 1]
                ]
                xs = bev_corners[..., 0] / bev_resolution + bev_w / 2
                ys = -bev_corners[..., 1] / bev_resolution + bev_h / 2
                for obj_idx, (x, y) in enumerate(zip(xs, ys)):
                    for p1, p2 in ((0, 1), (0, 2), (1, 3), (2, 3)):
                        if "track_ids" in result or "labels_3d" in result:
                            tmp = color[obj_idx]
                        else:
                            tmp = color
                        cv2.line(
                            bev,
                            (int(x[p1]), int(y[p1])),
                            (int(x[p2]), int(y[p2])),
                            tmp,
                            thickness=3,
                        )
            bev = np.uint8(bev)
            padding = np.zeros_like(bev)

            for j, name in enumerate(
                [
                    "front",
                    "front right",
                    "front left",
                    "rear",
                    "rear left",
                    "rear right",
                ]
            ):
                imgs[j] = cv2.rectangle(
                    imgs[j],
                    (0, 0),
                    (440, 80),
                    color=(255, 255, 255),
                    thickness=-1,
                )
                w, h = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
                text_x = int(220 - w / 2)
                text_y = int(40 + h / 2)

                imgs[j] = cv2.putText(
                    imgs[j],
                    name,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
            image = np.concatenate(
                [
                    np.concatenate([imgs[2], imgs[0], imgs[1]], axis=1),
                    np.concatenate([imgs[5], imgs[3], imgs[4]], axis=1),
                ],
                axis=0,
            )
            image = np.concatenate([image, bev], axis=1)
            if videoWriter is None:
                videoWriter = cv2.VideoWriter(
                    os.path.join(save_dir, "video.avi"),
                    fourcc,
                    7,
                    image.shape[:2][::-1],
                )
            cv2.imwrite(os.path.join(save_dir, f"{i}.jpg"), image)
            videoWriter.write(image)
        videoWriter.release()


def output_to_nusc_box(detection):
    box3d = detection["boxes_3d"]
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()
    if "track_ids" in detection:
        ids = detection["track_ids"]  # .numpy()

    if hasattr(box3d, "gravity_center"):
        box_gravity_center = box3d.gravity_center.numpy()
        box_dims = box3d.dims.numpy()
        nus_box_dims = box_dims[:, [1, 0, 2]]
        box_yaw = box3d.yaw.numpy()
    else:
        box3d = box3d.numpy()
        box_gravity_center = box3d[..., :3].copy()
        box_dims = box3d[..., 3:6].copy()
        nus_box_dims = box_dims[..., [1, 0, 2]]
        box_yaw = box3d[..., 6].copy()

    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    # box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if hasattr(box3d, "gravity_center"):
            velocity = (*box3d.tensor[i, 7:9], 0.0)
        else:
            velocity = (*box3d[i, 7:9], 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        if "track_ids" in detection:
            box.token = ids[i]
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(
    info,
    boxes,
    classes,
    eval_configs,
    eval_version="detection_cvpr_2019",
    sampled_points=None,
):
    box_list = []
    sampled_points_filtered = []
    for i, box in enumerate(boxes):
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info["lidar2ego_rotation"]))
        box.translate(np.array(info["lidar2ego_translation"]))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))
        box_list.append(box)
        if sampled_points is not None:
            sampled_points_filtered.append(sampled_points[i])
        else:
            sampled_points_filtered.append(None)
    return box_list, sampled_points_filtered


def plot_rect3d_on_img(
    img, num_rects, rect_corners, color=(0, 255, 0), thickness=1
):
    """Plot the boundary lines of 3D rectangular on 2D images.

    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    line_indices = (
        (0, 1),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 5),
        (3, 2),
        (3, 7),
        (4, 5),
        (4, 7),
        (2, 6),
        (5, 6),
        (6, 7),
    )
    h, w = img.shape[:2]
    for i in range(num_rects):
        corners = np.clip(rect_corners[i], -1e4, 1e5).astype(np.int32)
        for start, end in line_indices:
            if (
                (corners[start, 1] >= h or corners[start, 1] < 0)
                or (corners[start, 0] >= w or corners[start, 0] < 0)
            ) and (
                (corners[end, 1] >= h or corners[end, 1] < 0)
                or (corners[end, 0] >= w or corners[end, 0] < 0)
            ):
                continue
            if isinstance(color[0], int):
                cv2.line(
                    img,
                    (corners[start, 0], corners[start, 1]),
                    (corners[end, 0], corners[end, 1]),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
            else:
                cv2.line(
                    img,
                    (corners[start, 0], corners[start, 1]),
                    (corners[end, 0], corners[end, 1]),
                    color[i],
                    thickness,
                    cv2.LINE_AA,
                )

    return img.astype(np.uint8)


def draw_lidar_bbox3d_on_img(
    bboxes3d, raw_img, lidar2img_rt, img_metas, color=(0, 255, 0), thickness=1
):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`LiDARInstance3DBoxes`):
            3d bbox in lidar coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        img_metas (dict): Useless here.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    img = raw_img.copy()
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3), np.ones((num_bbox * 8, 1))], axis=-1
    )
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = pts_4d @ lidar2img_rt.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)


def draw_points_on_img(points, img, lidar2img_rt, color=(0, 255, 0), circle=4):
    img = img.copy()
    N = points.shape[0]
    points = points.cpu().numpy()
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = (
        np.sum(points[:, :, None] * lidar2img_rt[:3, :3], axis=-1)
        + lidar2img_rt[:3, 3]
    )
    pts_2d[..., 2] = np.clip(pts_2d[..., 2], a_min=1e-5, a_max=1e5)
    pts_2d = pts_2d[..., :2] / pts_2d[..., 2:3]
    pts_2d = np.clip(pts_2d, -1e4, 1e4).astype(np.int32)

    for i in range(N):
        for point in pts_2d[i]:
            if isinstance(color[0], int):
                color_tmp = color
            else:
                color_tmp = color[i]
            cv2.circle(img, point.tolist(), circle, color_tmp, thickness=-1)
    return img.astype(np.uint8)
