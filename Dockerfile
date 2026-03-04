FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Asia/Tokyo \
    DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda

ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates wget curl git build-essential \
      libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
      libgl1-mesa-dev language-pack-ja && \
    rm -rf /var/lib/apt/lists/*

# Install Miniforge + create env "app"
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm -f /tmp/miniforge.sh && \
    ${CONDA_DIR}/bin/conda config --system --set auto_activate_base false && \
    ${CONDA_DIR}/bin/conda config --system --prepend channels conda-forge && \
    ${CONDA_DIR}/bin/conda config --system --set channel_priority strict && \
    ${CONDA_DIR}/bin/conda create -y -n app python=3.10 numpy=1.26.4 pip && \
    ${CONDA_DIR}/bin/conda clean -afy && \
    test -d ${CONDA_DIR}/envs/app

WORKDIR /workspace
COPY install.sh /workspace/install.sh

# Run install.sh inside env "app"
RUN chmod +x /workspace/install.sh && \
    ${CONDA_DIR}/bin/conda run -n app bash /workspace/install.sh

CMD ["bash"]