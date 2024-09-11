#!/bin/bash
pwd

cd Training/vid2player3d

. /opt/conda/etc/profile.d/conda.sh

conda activate rlgpu

# pip install --force-reinstall --user "mujoco-py<2.2,>=2.1"

# pip uninstall -y numpy
# conda uninstall -y numpy

# conda install -y -c conda-forge numpy==1.23.5 --force

pip install "cython<3"

which python


echo $LD_LIBRARY_PATH
echo $LD_PRELOAD

export LD_LIBRARY_PATH=/opt/conda/envs/rlgpu/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/charlie/.mujoco/mujoco210/bin

python embodied_pose/run.py --cfg tennis_im --rl_device cuda:0 --headless

