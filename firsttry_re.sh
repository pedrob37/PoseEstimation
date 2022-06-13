#!/bin/bash

printf '%s\n' --------------------
echo APTGET
printf '%s\n' --------------------

#apt-get -y update
#apt-get install -y ffmpeg

printf '%s\n' --------------------
echo PIP
printf '%s\n' --------------------

pip install -r /nfs/home/ctangwiriyasakul/DGXwork/projects/2021/202110_Oct/week04/20211021_3DposeEST3D_MCA/requirements.txt

printf '%s\n' --------------------
echo PYTHON
printf '%s\n' --------------------

python3 "/nfs/home/ctangwiriyasakul/DGXwork/projects/2021/202110_Oct/week04/20211021_3DposeEST3D_MCA/main_training_01_v2_resize.py"
