FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:0.5.30 /uv /usr/local/bin/uv
COPY --from=ghcr.io/astral-sh/uv:0.5.30 /uvx /usr/local/bin/uvx

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV UV_NO_SYNC=1
ENV UV_LINK_MODE=copy
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV WANDB_MODE=disabled
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV PYTHONPATH=/workspace/openvla:/workspace/LIBERO:/opt/src/openvla:/opt/src/LIBERO

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libglvnd0 \
    libglx0 \
    libosmesa6 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ninja-build \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv /opt/venv --python /usr/bin/python3.10

RUN uv pip install --python /opt/venv/bin/python \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1

RUN uv pip install --python /opt/venv/bin/python \
    accelerate==0.25.0 \
    "bddl==1.0.1" \
    cloudpickle \
    draccus==0.8.0 \
    easydict \
    einops \
    future==0.18.2 \
    "gym==0.25.2" \
    "huggingface_hub" \
    "imageio[ffmpeg]" \
    json-numpy \
    jsonlines \
    matplotlib \
    mujoco==2.3.7 \
    "numpy<2" \
    opencv-python-headless \
    packaging \
    peft==0.11.1 \
    protobuf==3.20.3 \
    rich \
    robosuite==1.4.1 \
    sentencepiece==0.1.99 \
    tensorflow==2.15.0 \
    tensorflow_datasets==4.9.3 \
    tensorflow_graphics==2021.12.3 \
    tensorflow-metadata==1.14.0 \
    timm==0.9.10 \
    tokenizers==0.19.1 \
    transformers==4.40.1 \
    wandb

RUN uv pip install --python /opt/venv/bin/python \
    "dlimp @ git+https://github.com/moojink/dlimp_openvla@040105d256bd28866cc6620621a3d5f7b6b91b46"

COPY openvla /opt/src/openvla
COPY LIBERO /opt/src/LIBERO

RUN uv pip install --python /opt/venv/bin/python --no-deps -e /opt/src/openvla -e /opt/src/LIBERO

WORKDIR /workspace
CMD ["bash"]
