## Preparing Tennis Dataset

1. Get Broadcast videos (i.e. youtube 'full tennis match')
2. Install dependencies for preparing the dataset
   **Note: Dependencies can be found in their respective original github repos, but there are some changes I have made to file-structure/altered files to fit the purpose.**
    - Install all in `atpil.yml`
    - Install Hybrik from the submodule into same environment.
    - Install Spot from submodule
    - Install tennis-court-detector (requires building. suggest visiting that repo.)
    - `conda activate atpil`
4. Run `python prepare_dataset.py` (see file for required parameters)

This will: 
- Extract tennis playing sequences into the provided output file (motion library)
- Make a `manifest.json` describing the sequences. **Note: all sequences are annotated as 'kyrgios', 'right handed', and 'eastern' grip. Correct annotation can be done after or incorporated as future work.**

## Train the Embodied Pose Imitation Network

Everything from here is done in the `Training` folder.

1. Train using AMASS data. AMASS data first needs to be converted with `convert_amass_isaac.py` (in vid2player3d/uhc/utils)
2. Edit/use embodied_pose/cfg/amass_im to point to the correct motion_lib
3. Train using the prepared tennis data, transfer learning from the amass model.
4. Edit/use embodied_pose/cfg/tennis_im, or train a specific model with a specific player's information.

## Train the MVAE for motion generation

1. Need to export all of the embodied pose tracked motions.
2. I don't know how this was originally done, but using --test --export_dataset <folder name> is what I implemented. Sometimes you need to look into the number of games played (embodied_pose/players) to make sure all motions are tracked.
3. This should output a heap of tracked motions.
4. join_mvae_dataset.py will join the tracked motions into the master file required. Look in there for parameters.
5. Train with Training/vid2player3d/vid2player/motion_vae code. There are configs and stuff that need to be worked out.

## Train the player
If all of the above is done, go to Training/vid2player3d/vid2player/cfg and make sure there's a config that makes sense. 3 stages seems to work well so copy one of those.

Train with `vid2player/run.py --cfg <your_config>`

------- Road-map / checklist -----------

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
