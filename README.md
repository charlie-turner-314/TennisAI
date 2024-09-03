## Build EmbodiedPose Dataset

> You first must obtain videos of tennis from the 'broadcast' camera angle (i.e. from behind the baseline). The AMASS dataset is also required

- [x] CropVideo - Crops to the player in the 'near' court
- [x] PoseDetection2d - Detects the player's 2d key-points
- [x] PoseEstimation3D - Estimates the player's 3d pose  

## Build MVAE Dataset

- [x] tennis-court-detection - Detects the tennis court lines
- [x] use the tennis court lines to position player in court coordinates
- [x] export EmbodiedPose data in court coordinates
- [x] spot - Detects the ball contact points

All of this is contained within `prepare_dataset.py`.

## Train 
- [x] Train EmbodiedPose - Imitation Learning Model
  - [x] EmbodiedPose - AMASS
  - [x] EmbodiedPose - Tennis

> The EmbodiedPose model looks pretty good!

- [x] Train MVAE - Variational Autoencoder to Generate novel motions

> This looks okay too

- [x] Train vid2player3d - High level model to respond to incoming balls

> All of this works but doesn't produce great models
