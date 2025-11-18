FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

ENV FORCE_CUDA="1"
ENV MMCV_WITH_OPS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip     \
    libgl1-mesa-glx \
    libsm6          \
    libxext6        \
    libxrender-dev  \
    libglib2.0-0    \
    git             \
    python3-dev     \
    python3-wheel   \
    curl            \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip \
    && pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu117 \
        wheel \
        "torch>=2.0.2" \
        "torchvision>=0.15.2" \
        "torchaudio>=2.0.2" \
    && pip3 install --no-cache-dir \
        "numpy>=1.26.0,<2.0.0" \
        "scipy>=1.15.0" \
        "matplotlib>=3.10.0" \
        "pillow>=11.0.0" \
        "pyyaml>=6.0.0" \
        "requests>=2.28.0" \
        "tqdm>=4.65.0" \
        setuptools \
    && pip3 install --no-cache-dir mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html \
    && pip3 install --no-cache-dir \
        "mmengine==0.10.0" \
        "mmdet==3.0.0" \
        "mmyolo==0.6.0" \
        "opencv-python>=4.8.0" \
        "opencv-python-headless>=4.9.0" \
        "albumentations>=2.0.0" \
        "timm>=0.6.0" \
        "transformers>=4.36.0" \
        "tokenizers>=0.15.0" \
        "huggingface-hub>=0.36.0" \
        "safetensors>=0.6.0" \
        "pycocotools>=2.0.0" \
        "lvis>=0.5.3" \
        "shapely>=2.0.0" \
        "rich>=14.0.0" \
        "prettytable>=3.16.0" \
        "yapf>=0.43.0" \
        "addict>=2.4.0" \
        "termcolor>=3.2.0" \
        "terminaltables>=3.1.0"

COPY . /yolo-world
WORKDIR /yolo-world

RUN pip3 install --no-cache-dir -e .

RUN mkdir -p weights && \
    curl -o weights/yolo_world_v2_s_stage1.pth -L https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/s_stage1-4466ab94.pth && \
    curl -o weights/yolo_world_v2_s_stage2.pth -L https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/s_stage2-4466ab94.pth && \
    curl -o weights/yolo_world_v2_m_stage1.pth -L https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/m_stage1-7e1e5299.pth && \
    curl -o weights/yolo_world_v2_m_stage2.pth -L https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/m_stage2-9987dcb1.pth && \
    curl -o weights/yolo_world_v2_l_stage1.pth -L https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/l_stage1-7d280586.pth && \
    curl -o weights/yolo_world_v2_l_stage2.pth -L https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/l_stage2-b3e3dc3f.pth && \
    curl -o weights/yolo_world_v2_x_stage1.pth -L https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/x_stage1-62b674ad.pth

ENTRYPOINT ["PYTHONPATH=./ python3", "inference/inference.py"]
CMD ["--config", "configs/inference/yolo_world_v2_x_inference.yaml"]