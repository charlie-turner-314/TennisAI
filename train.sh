#!/bin/bash

micromamba run pip install -e Training/smpl_visualizer

cd Training/vid2player3d

micromamba run python vid2player/run.py --cfg kyrgios_train_stage_1 --headless --rl_device cuda:0 --resume --checkpoint latest
