FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Asia/Tokyo \
    DEBIAN_FRONTEND=noninteractive

# openssh-client や OpenCV用のライブラリ を追加
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates wget curl git build-essential openssh-client \
      libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
      libgl1-mesa-dev libglib2.0-0 libsm6 libxext6 libxrender-dev language-pack-ja && \
    rm -rf /var/lib/apt/lists/*

# Install Miniforge + create env
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm -f /tmp/miniforge.sh && \
    ${CONDA_DIR}/bin/conda config --system --set auto_activate_base false && \
    ${CONDA_DIR}/bin/conda config --system --prepend channels conda-forge && \
    ${CONDA_DIR}/bin/conda config --system --set channel_priority strict && \
    ${CONDA_DIR}/bin/conda create -y -n env_dms python=3.10 && \
    ${CONDA_DIR}/bin/conda clean -afy && \
    test -d ${CONDA_DIR}/envs/env_dms

WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt

# コンテナに入った瞬間から env_dms 環境にする
ENV PATH="${CONDA_DIR}/envs/env_dms/bin:${PATH}"

# ▼▼ ここを追加！ MMCVのビルドに必須のCUDA環境変数 ▼▼
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9;12.0"

# ==============================================================================
# 依存関係と OpenMMLab ライブラリの確実なビルドプロセス
# ==============================================================================
# 1. git+ で始まる行を除外し、PyTorchのNightly版URLを指定してベース要件をインストール
# 2. setuptools をダウングレード (pkg_resources エラー回避)
# 3. マウント(-v)で消えないように /opt/mmlab に隔離して MMCV と MMDetection をビルド
RUN mkdir -p /opt/mmlab && \
    grep -v "git+https" /workspace/requirements.txt > /tmp/req_base.txt && \
    pip install -r /tmp/req_base.txt --extra-index-url https://download.pytorch.org/whl/nightly/cu128 && \
    pip install "setuptools<70" --force-reinstall && \
    cd /opt/mmlab && \
    env HOME=/tmp GIT_CONFIG_NOSYSTEM=1 git clone https://github.com/open-mmlab/mmcv.git && \
    cd mmcv && \
    git checkout 57c4e25e06e2d4f8a9357c84bcd24089a284dc88 && \
    pip install . -v --no-build-isolation && \
    cd /opt/mmlab && \
    env HOME=/tmp GIT_CONFIG_NOSYSTEM=1 git clone https://github.com/open-mmlab/mmdetection.git && \
    cd mmdetection && \
    git checkout cfd5d3a985b0249de009b67d04f37263e11cdf3d && \
    pip install . -v --no-build-isolation && \
    rm -rf /tmp/req_base.txt

CMD ["bash"]