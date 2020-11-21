# System Environment
- ubuntu 16.04
- python 3.7.9
- cuda 10.1

# Requirements file
requirements/build.txt

# Install
```bash
conda create -n dacon-mmlab python=3.7.9 -y
conda activate dacon-mmlab

conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -y

pip install mmcv-full==1.1.2+torch1.6.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html

git clone https://github.com/jireh-father/mmdetection_for_dacon_kfashion_ai.git

cd mmdetection_for_dacon_kfashion_ai

pip install -r requirements/build.txt
pip install -v -e .
```

# Training, validation and submission
위 환경에서 main.ipynb 를 실행하여 순서대로 실행

## 솔루션 설명 PPT
https://drive.google.com/file/d/1l_TrnCSlYzDLmd_wh68YXHoz6eHEg6bK/view?usp=sharing
