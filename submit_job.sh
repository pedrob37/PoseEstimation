#!/bin/bash

runai submit train03 \
  --image 10.202.67.207:5000/ctangwiriyasakul:junction2 \
  --backoffLimit 0 \
  --volume /nfs:/nfs \
  --gpu 1 \
  --memory-limit 60G \
  --large-shm \
  --project ctangwiriyasakul \
  --command /nfs/home/ctangwiriyasakul/DGXwork/projects/2021/202110_Oct/week04/20211021_3DposeEST3D_MCA/firsttry.sh  \
  --run-as-user
