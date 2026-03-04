#!/bin/bash
set -euo pipefail
# # create venv
# python -m venv .venv_DeMemSeg
# source .venv_DeMemSeg/bin/activate

python -V
python -c "import numpy as np; print('numpy', np.__version__, np.__file__)"
python -m pip -V


# install dependencies
python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -U openmim
python -m pip install mmengine
python -m pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
python -m pip install -v . --no-build-isolation

python -m pip install scikit-image tifffile tensorboard seaborn

python -m pip install cellpose==3.1.0
python -m pip install --force-reinstall "numpy==1.26.4"