#!/bin/bash
set -euo pipefail

ENV=app

# まず環境が正しいか確認
micromamba run -n $ENV python -V
micromamba run -n $ENV python -c "import numpy as np; print('numpy', np.__version__, np.__file__)"

# pip 基本
micromamba run -n $ENV python -m pip install -U pip setuptools==60.2.0 wheel

# 1) torch/cu118 (先に入れる)
micromamba run -n $ENV python -m pip install \
  torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 2) OpenMMLab (mmengine/mmcv)
micromamba run -n $ENV python -m pip install -U mmengine
micromamba run -n $ENV python -m pip install \
  mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

# 3) mmdetection (editable / pep517回避)
if [ ! -d /workspace/mmdetection ]; then
  git clone https://github.com/open-mmlab/mmdetection.git /workspace/mmdetection
fi

micromamba run -n $ENV python -m pip install -v -e /workspace/mmdetection \

# 4) cellpose（numpy 1.26 で動かす）
micromamba run -n $ENV python -m pip install \
  scikit-image tifffile tensorboard seaborn cellpose==3.1.0 numpy==1.26

# 仕上げ：依存の確認
micromamba run -n $ENV python -c "import torch; import numpy as np; print('torch', torch.__version__); print('numpy', np.__version__)"
micromamba run -n $ENV python -c "import mmcv; import mmdet; print('mmcv', mmcv.__version__); print('mmdet', mmdet.__version__)"
micromamba run -n $ENV python -c "from cellpose import models; print('cellpose ok')"