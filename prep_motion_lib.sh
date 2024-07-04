#!/bin/bash

cd Training
conda activate rlgpu
python preprocess/preprocess_tennis.py --input_dir ../data/meshes/translated --output ./vid2player3d/data/tennis/processed_translated.pkl
cd vid2player3d
python uhc/utils/convert_amass_isaac.py --amass_data ./data/tennis/processed_translated.pkl --out_dir ./data/motion_lib/tennis_trans --num_motion_libs 1