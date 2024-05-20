# DDet3D: Embracing 3D Object Detector with Diffusion

This is the official PyTorch implementation of the paper **DDet3D: Embracing 3D Object Detector with Diffusion**, by Gopi Krishna Erabati and Helder Araujo.

**Contents**
1. [Overview](https://github.com/gopi-erabati/DDet3D#overview)
2. [Results](https://github.com/gopi-erabati/DDet3D#results)
3. [Requirements, Installation and Usage](https://github.com/gopi-erabati/DDet3D#requirements-installation-and-usage)
    1. [Prerequistes](https://github.com/gopi-erabati/DDet3D#prerequisites)
    2. [Installation](https://github.com/gopi-erabati/DDet3D#installation)
    3. [Training](https://github.com/gopi-erabati/DDet3D#training)
    4. [Testing](https://github.com/gopi-erabati/DDet3D#testing)
4. [Acknowledgements](https://github.com/gopi-erabati/DDet3D#acknowlegements)
5. [Reference](https://github.com/gopi-erabati/DDet3D#reference)

## Overview
Existing approaches rely on heuristic or learnable object proposals (which are required to be optimised during training) for 3D object detection. In our approach, we replace the hand-crafted or learnable object proposals with random object proposals by formulating a new paradigm to employ diffusion model to detect 3D objects from a set of random object proposals in an autonomous driving application. We propose DDet3D, a diffusion based 3D object detection framework that formulates 3D object detection as a generative task over the 3D bounding box coordinates in the 3D space. To our knowledge, this work is the first to formulate the 3D object detection with denoising diffusion model and to establish that 3D random proposals (different from empirical anchors or learnt queries) are also potential object candidates for 3D object detection. During training, the 3D random noisy boxes are employed from the 3D ground truth boxes by progressively adding Gaussian noise, and the DDet3D network is trained to reverse the diffusion process. During the inference stage, DDet3D network is able to iteratively refine the 3D random noisy boxes to predict 3D bounding boxes conditioned on the LiDAR Bird’s Eye View (BEV) features. The advantage of DDet3D is that it allows to decouple training and inference stages, thus enabling to employ more number of proposal boxes or sampling steps during inference to improve the accuracy. We conduct extensive experiments and analysis on nuScenes and KITTI datasets. DDet3D achieves competitive performance compared to well-designed 3D object detectors. Our work serves as a strong baseline to explore and employ more efficient diffusion models to the 3D perception tasks.

![ddet3d(1)](https://github.com/gopi-erabati/DDet3D/assets/22390149/7657f2b9-44be-421f-b94f-bf771ea646f0)

## Results

### Predictions on nuScenes dataset
![DDet3D_Diffusionbased3DObjectDetection_3DObjectDetection_AutonomousDriving_Diffusion-ezgif com-video-to-gif-converter](https://github.com/gopi-erabati/DDet3D/assets/22390149/f042ae1c-90d8-4df7-a1cf-65f964a2ca09)

### Quantitative Results
| Config | steps| mAP | NDS | |
| :---: | :---: |:---: |:---: |:---: |
| ddet3d_voxel_nus.py | 1 | 61.7 | 65.9 | [weights](https://drive.google.com/file/d/18QQQChBIulw_s0o2Hbks38YK_f__Xqtp/view?usp=sharing) |
| ddet3d_voxel_nus.py | 2 | 62.9 | 66.4 | |
| ddet3d_voxel_nus.py | 4 | 63.2 | 66.6 | |

## Requirements, Installation and Usage

### Prerequisites

The code is tested on the following configuration:
- Ubuntu 20.04.6 LTS
- CUDA==11.7
- Python==3.8.10
- PyTorch==1.13.1
- [mmcv](https://github.com/open-mmlab/mmcv)==1.7.0
- [mmdet](https://github.com/open-mmlab/mmdetection)==2.28.2
- [mmseg](https://github.com/open-mmlab/mmsegmentation)==0.30.0
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)==1.0.0.rc6

### Installation
```
mkvirtualenv ddet3d

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -U openmim
mim install mmcv-full==1.7.0

pip install -r requirements.txt
```

### Data
Follow [MMDetection3D-1.0.0.rc6](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0rc6) to prepare the [nuScenes](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/nuscenes.html) dataset and symlink the data directory to `data/` folder of this repository.

**Warning:** Please strictly follow [MMDetection3D-1.0.0.rc6](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0rc6) code to prepare the data because other versions of MMDetection3D have different coordinate refactoring.

### Clone the repository
```
git clone https://github.com/gopi-erabati/DDet3D.git
cd DDet3D
```

### Training

- Download the backbone pretrained weights to `ckpts/`
- Single GPU training
    1. Add the present working directory to PYTHONPATH `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/train.py configs/ddet3d_voxel_nus.py --work-dir {WORK_DIR}`.
- Multi GPU training
  `tools/dist_train.sh configs/ddet3d_voxel_nus.py {GPU_NUM} --work-dir {WORK_DIR}`.

### Testing

- Single GPU testing
    1. Add the present working directory to PYTHONPATH `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/test.py configs/ddet3d_voxel_nus.py /path/to/ckpt --eval mAP`.
- Multi GPU testing  `tools/dist_test.sh configs/ddet3d_voxel_nus.py /path/to/ckpt {GPU_NUM} --eval mAP`.

## Acknowlegements
We sincerely thank the contributors for their open-source code: [MMCV](https://github.com/open-mmlab/mmcv), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMDetection3D](https://github.com/open-mmlab/mmdetection3d).

## Reference
```

```
