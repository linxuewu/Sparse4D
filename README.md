# Sparse4D

## Overall Architecture
<img src="resources/framework.png" width="1000" >
[video demo](https://github.com/linxuewu/Sparse4D/releases/download/v0.0/video.avi)

## nuScenes Benchmark
|config |mAP |mATE| mASE | mAOE |mAVE| mAAE | NDS | checkpoint | 
| ------ | :---:|:---:|:---:  | :---: | :----: | :----: | :----: | :----: |
|[R101-T4](https://github.com/linxuewu/Sparse4D/releases/download/v0.0/sparse4d_det_resnet101.log)|0.4360| 0.6325| 0.2787| 0.3626| 0.3166| 0.1765| 0.5413 | [ckpt](https://github.com/linxuewu/Sparse4D/releases/download/v0.0/sparse4d_det_resnet101.pth) |

## Quick Start

 Install requirements.
```shell
pip install -r requirements.txt
```

Download nuScenes dataset, pretrain checkpoint([fcos3d.pth ResNet101](https://github.com/linxuewu/Sparse4D/releases/download/v0.0/fcos3d.pth)), pkl files([nuscenes_infos_trainval_with_inds.pkl](https://github.com/linxuewu/Sparse4D/releases/download/v0.0/nuscenes_infos_trainval_with_inds.pkl)) and init anchor centers([nuscenes_centers_kmeans900.npy](https://github.com/linxuewu/Sparse4D/releases/download/v0.0/nuscenes_centers_kmeans900.npy)). Adjust the directory structure as follows:
```shell
Sparse4D
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── lidarseg
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-mini
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   ├── nuscenes_cam
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_trainval_with_inds.pkl
├── projects
│   ├── configs
│   │   ├── default_runtime.py
│   │   ├── sparse4d_det_resnet101.py
│   │   ├── ...
│   └── mmdet3d_plugin
│       ├── apis
│       ├── core
│       ├── datasets
│       ├── models
├── tools
│   ├── dist_test.sh
│   ├── dist_train.sh
│   ├── test.py
│   └── train.py
├── local_test.sh
├── local_train.sh
├── fcos3d.pth
└── nuscenes_centers_kmeans900.npy
```

Train with config_name.py.
```shell
bash local_train.sh config_name
```

Test checkpoint_file with config_name.py.
```shell
bash local_test.sh config_name checkpoint_file
```
