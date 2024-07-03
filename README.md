## Build EmbodiedPose Dataset

> You first must obtain videos of tennis from the 'broadcast' camera angle (i.e. from behind the baseline). The AMASS dataset is also required

- [x] CropVideo - Crops to the player in the 'near' court
- [x] PoseDetection2d - Detects the player's 2d key-points
- [x] PoseEstimation3D - Estimates the player's 3d pose
- [x] Training
  - [x] EmbodiedPose - AMASS
  - [x] EmbodiedPose - Tennis

## Build MVAE Dataset

- [x] tennis-court-detection - Detects the tennis court lines
- [ ] use the tennis court lines to position player in court coordinates
- [ ] export EmbodiedPose data in court coordinates
- [ ] label frames where the ball is hit
