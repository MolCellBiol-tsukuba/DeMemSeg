docker build -t dememseg-env .

docker run -it --name dememseg --gpus all --rm -v "$(pwd)":/workspace dememseg-env

conda minicondaとかインストールする必要ないかも



cellposeのインストール
pip install cellpose==3.1.0

pip install scikit-image tifffile tensorboard seaborn

mmdetectionのインストール
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118



pip install -U openmim
pip install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install -v -e .
git clone https://github.com/open-mmlab/mmdetection.git
pip install scikit-image tifffile tensorboard seaborn




bash install.shのあつかいをdockerfileに追加する。