#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

if [ ! -d YCB_Video_toolbox ];then
    echo 'Downloading the YCB_Video_toolbox...'
    git clone https://github.com/yuxng/YCB_Video_toolbox.git
    cd YCB_Video_toolbox
    unzip results_PoseCNN_RSS2018.zip
    cd ..
    cp replace_ycb_toolbox/*.m YCB_Video_toolbox/
fi

python ./tools/eval_ycb.py --dataset_root ./datasets/ycb/YCB_Video_Dataset\
  --model pose_model_26_0.012863246640872631.pth\
  --refine_model pose_refine_model_69_0.009449292959118935.pth