<div align="center">
<img src="./assets/yolo_logo.png" width=60%>
<br>

[![arxiv paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2401.17270)
[![demo](https://img.shields.io/badge/🤗HuggingFace-Demo-orange)](https://huggingface.co/spaces/stevengrove/YOLO-World)
[![license](https://img.shields.io/badge/License-GPLv3.0-blue)](LICENSE)

**Real-Time Open-Vocabulary Object Detection**

</div>

## Table of Contents
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Inference](#inference)
- [Pre-trained Models](#pre-trained-models)
- [Training & Fine-tuning](#training--fine-tuning)
- [Advanced](#advanced)

---

## Quick Start

**Installation (3 commands):**
```bash
git clone --recursive https://github.com/ArqumUddin/YOLO-World.git
cd YOLO-World
./setup.sh
```

**Run Inference:**
```bash
conda activate yolo-world
python -m inference --config configs/inference/coco_classes.yaml
```

**Start API Server:**
```bash
python -m inference.server \
    --config configs/pretrain/yolo_world_v2_l.py \
    --checkpoint weights/yolo_world_v2_l_stage1.pth \
    --prompts "person,car,dog,cat"
```

---

## Installation

### Requirements
- Python 3.10
- CUDA 11.7/11.8
- Conda or Miniconda

### Automated Setup

```bash
git clone --recursive https://github.com/ArqumUddin/YOLO-World.git
cd YOLO-World
chmod +x setup.sh
./setup.sh
conda activate yolo-world
```

The setup script installs:
- PyTorch 2.0 with CUDA support
- MMDetection 3.0.0 & MMYOLO 0.6.0
- Pre-trained weights (S, M, L, X models)
- All dependencies

**Verify:**
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import yolo_world; print('✓ YOLO-World ready')"
```

---

## Inference

### Local Inference

Run detection on images or videos:

```bash
python -m inference --config configs/inference/coco_classes.yaml
```

**Config Example** (`configs/inference/your_config.yaml`):
```yaml
model:
  config: "configs/pretrain/yolo_world_v2_l.py"
  checkpoint: "weights/yolo_world_v2_l_stage1.pth"

detection:
  text_prompts: ["person", "car", "dog", "cat"]
  confidence_threshold: 0.05

output:
  save_annotated: true
```

**Output:**
- Annotated video/frames with bounding boxes
- `results.json` with detections, metrics, and statistics

### REST API Server

**Start Server:**
```bash
python -m inference.server \
    --config configs/pretrain/yolo_world_v2_l.py \
    --checkpoint weights/yolo_world_v2_l_stage1.pth \
    --port 12182 \
    --prompts "person,car,dog,cat,bird"
```

**Client Code:**
```python
import base64
import requests
import cv2

# Encode image
img = cv2.imread("image.jpg")
_, buffer = cv2.imencode(".jpg", img)
img_str = base64.b64encode(buffer).decode("utf-8")

# Request
response = requests.post(
    "http://localhost:12182/yolo_world",
    json={"image": img_str, "caption": "person . car . dog ."}
)

print(f"Found {response.json()['num_detections']} objects")
```

**Server Options:**
- `--port`: Port number (default: 12182)
- `--device`: cuda/cpu (auto-detect)
- `--confidence`: Threshold (default: 0.05)
- `--nms`: NMS threshold (default: 0.7)

---

## Pre-trained Models

### Model Downloads

All models are automatically downloaded by `setup.sh`. Manual downloads available from [HuggingFace](https://huggingface.co/wondervictor/YOLO-World-V2.1).

| Model | Resolution | LVIS AP | COCO AP | Weights |
|-------|-----------|---------|---------|---------|
| YOLO-World-S | 640 | 18.5 | 36.6 | [Download](https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/s_stage1-4466ab94.pth) |
| YOLO-World-M | 640 | 24.1 | 43.0 | [Download](https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/m_stage1-7e1e5299.pth) |
| YOLO-World-L | 640 | 26.8 | 44.9 | [Download](https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/l_stage1-7d280586.pth) |
| YOLO-World-X | 640 | 28.6 | 46.7 | [Download](https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/x_stage1-62b674ad.pth) |

### Key Features

- **Open-Vocabulary Detection**: Detect any object class by text description
- **Real-Time Performance**: Fast inference on GPU
- **Zero-Shot Capability**: No training required for new classes
- **Flexible Deployment**: Local, server, or cloud-based

---

## Training & Fine-tuning

### Pre-training

```bash
./tools/dist_train.sh configs/pretrain/yolo_world_l_[config].py 8 --amp
```

### Evaluation

```bash
./tools/dist_test.sh path/to/config path/to/weights 8
```

### Fine-tuning Options

1. **Normal Fine-tuning**: [docs/finetuning.md](./docs/finetuning.md)
2. **Prompt Tuning**: [docs/prompt_yolo_world.md](./docs/prompt_yolo_world.md)
3. **Reparameterized Fine-tuning**: [docs/reparameterize.md](./docs/reparameterize.md)

---

## Advanced

### Documentation
- [Data Preparation](./docs/data.md)
- [Configuration Guide](./docs/)
- [FAQ](https://github.com/AILab-CVC/YOLO-World/discussions/149)
- [Roadmap](https://github.com/AILab-CVC/YOLO-World/issues/109)

### Integrations
- [HuggingFace Spaces](https://huggingface.co/spaces/stevengrove/YOLO-World)
- [ComfyUI](https://github.com/StevenGrove/ComfyUI-YOLOWorld)
- [FiftyOne](https://docs.voxel51.com/integrations/ultralytics.html)
- [Roboflow](https://inference.roboflow.com/foundation/yolo_world/)

### Updates

**Latest (2025-02-08):** YOLO-World V2.1 released with improved weights and image prompt support. See [update notes](./docs/update_20250123.md).

**CVPR 2024:** Paper accepted at Computer Vision and Pattern Recognition 2024.

---

## Citation

```bibtex
@inproceedings{Cheng2024YOLOWorld,
  title={YOLO-World: Real-Time Open-Vocabulary Object Detection},
  author={Cheng, Tianheng and Song, Lin and Ge, Yixiao and Liu, Wenyu and Wang, Xinggang and Shan, Ying},
  booktitle={Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

## Acknowledgements

Thanks to [mmyolo](https://github.com/open-mmlab/mmyolo), [mmdetection](https://github.com/open-mmlab/mmdetection), [GLIP](https://github.com/microsoft/GLIP), and [transformers](https://github.com/huggingface/transformers).

## License

YOLO-World is under GPL-v3 License. For commercial licensing, contact `yixiaoge@tencent.com`.

---

<div align="center">

**[Project Page](https://wondervictor.github.io/)** | **[Paper](https://arxiv.org/abs/2401.17270)** | **[Demo](https://huggingface.co/spaces/stevengrove/YOLO-World)**

</div>
