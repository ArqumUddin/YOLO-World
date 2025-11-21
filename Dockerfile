FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV FORCE_CUDA="1"
ENV MMCV_WITH_OPS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libsm6          \
    libxext6        \
    libxrender-dev  \
    libglib2.0-0    \
    git             \
    curl            \
    wget            \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

ENV PATH="/opt/conda/bin:${PATH}"

# Copy environment file
COPY environment.yml /tmp/environment.yml

# Create conda environment from environment.yml
RUN conda env create -f /tmp/environment.yml \
    && conda clean -afy \
    && rm /tmp/environment.yml

# Activate conda environment
ENV PATH="/opt/conda/envs/yolo-world/bin:${PATH}"
ENV CONDA_DEFAULT_ENV=yolo-world

COPY . /yolo-world
WORKDIR /yolo-world

RUN mkdir -p weights && \
    curl -o weights/yolo_world_v2_s_stage1.pth -L https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/s_stage1-4466ab94.pth && \
    curl -o weights/yolo_world_v2_s_stage2.pth -L https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/s_stage2-4466ab94.pth && \
    curl -o weights/yolo_world_v2_m_stage1.pth -L https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/m_stage1-7e1e5299.pth && \
    curl -o weights/yolo_world_v2_m_stage2.pth -L https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/m_stage2-9987dcb1.pth && \
    curl -o weights/yolo_world_v2_l_stage1.pth -L https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/l_stage1-7d280586.pth && \
    curl -o weights/yolo_world_v2_l_stage2.pth -L https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/l_stage2-b3e3dc3f.pth && \
    curl -o weights/yolo_world_v2_x_stage1.pth -L https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/x_stage1-62b674ad.pth

ENTRYPOINT ["python", "inference/inference.py"]
CMD ["--config", "configs/inference/yolo_world_v2_x_inference.yaml"]