#!/bin/bash

source activate base

# Create the conda environment
conda env create -f atpil.yml
conda activate atpil

# Install HybrIK stuff
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
cd dataset_preparation/estimate_pose_3d/HybrIK
python setup.py develop
cd ../../..

# Install vid2player3d stuff
cd Training/vid2player3d
./install.sh
pip install lxml joblib numpy-stl
pip install -U 'mujoco-py<2.2,>=2.1'
