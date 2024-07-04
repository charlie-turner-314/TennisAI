import argparse

import numpy as np
import os
import cv2
import json
import time
from dataset_preparation.player_detection.player_detect import detect_player
from dataset_preparation.player_detection.crop_video import crop_video
from dataset_preparation.estimate_pose.estimate_pose import detect_pose
from Training.vid2player3d.process_hybrik_data import process_hybrik


class COLORS:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    ENDC = "\033[0m"


parser = argparse.ArgumentParser()

parser.description = "Prepare tennis dataset for training."

parser.add_argument(
    "--input_dir",
    type=str,
    default="data/full_clips",
    help="Directory with mp4 tennis clips from broadcast angle. Each video should be a single shot sequence.",
    dest="input_dir",
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="data/processed",
    help="Directory to save processed clips.",
    dest="output_dir",
)

args = parser.parse_args()

print("Preparing dataset using videos from: ", args.input_dir)
print("Outputs will be saved to: ", args.output_dir)
time.sleep(2)

all_videos = os.listdir(args.input_dir)
all_videos = [v for v in all_videos if v.endswith(".mp4")]
aux_dir = os.path.join(args.input_dir, "..", "aux")

for video_name in all_videos:
    video_path = os.path.join(args.input_dir, video_name)
    video_name = video_name.split(".")[0]
    print(COLORS.BLUE + f"Processing video: {video_path}" + COLORS.ENDC)
    # ======================
    # Crop To Player
    # ======================
    # player bounding box for each frame
    # (bbox_w, bbox_h), positions = detect_player(filepath=video, player="fg")
    # # Crop video to player bounding box
    # success, out_file = crop_video(
    #     video, positions, bbox_w, bbox_h, os.path.join(aux_dir, "cropped")
    # )
    # if not success:
    #     print(f"Failed to crop video {video}.")
    #     continue
    # print(COLORS.GREEN + f"Saved cropped video to: {out_file}" + COLORS.ENDC)
    # # ======================
    # # 2D Pose Estimation
    # # ======================
    # detect_pose(
    #     video=out_file,
    #     out_dir=os.path.join(aux_dir, "pose"),
    #     save_video=True,
    #     device="cpu",
    # )
    # print(
    #     COLORS.GREEN
    #     + f"Saved pose estimation to: {os.path.join(aux_dir, 'pose')}"
    #     + COLORS.ENDC
    # )
    # ======================
    # 3D Mesh Estimation
    # ======================
    # out_file = os.path.join(aux_dir, "cropped", "federer_front.mp4")
    # out_file = os.path.abspath(out_file)
    # out_dir = os.path.join(aux_dir, "pose_3d")
    # out_dir = os.path.abspath(out_dir)

    # os.chdir("dataset_preparation/estimate_pose_3d/HybrIK")
    # from dataset_preparation.estimate_pose_3d.estimate_pose import estimate_pose_3d

    # estimate_pose_3d(
    #     video_name=out_file,
    #     out_dir=out_dir,
    #     save_video=False,
    #     device="cpu",
    # )
    # print(
    #     COLORS.GREEN
    #     + f"Saved 3D pose estimation to: {os.path.join(aux_dir, 'pose_3d')}"
    #     + COLORS.ENDC
    # )
    # os.chdir("../../..")
    # ======================
    # Tennis Court Line Detection
    # ======================
    # TODO: this takes a while -> perhaps a tag in the filename that says which long-form video it came from, then only do that once.
    # step 1: Convert the video to an avi with ffmpeg
    infile = video_path
    avi_file = os.path.join(aux_dir, "avi", "video.avi")
    os.makedirs(os.path.dirname(avi_file), exist_ok=True)
    command = f"ffmpeg -i {infile} -vcodec mjpeg -q:v 3 -acodec pcm_s16le {avi_file} -loglevel quiet -y"
    os.system(command)
    # Run the c++ application to detect tennis court lines in the video
    lines_file = os.path.join(aux_dir, "lines", f"{video_name}.txt")
    os.makedirs(os.path.dirname(lines_file), exist_ok=True)
    command = f"dataset_preparation/court_detection/tennis-court-detection/build/bin/detect {avi_file} {lines_file}"
    os.system(command)
    # ======================
    # Process HybrIK Data (Rotate to global frame)
    # ======================
    hybrik_file = os.path.join(aux_dir, "pose_3d", f"res_{video_name}.pk")
    output_file = os.path.join(aux_dir, "pose_3d", "processed", f"{video_name}.pkl")
    process_hybrik(input_file=hybrik_file, output_file=output_file)
    # ======================
    # Convert root position and orientation to court coordinates
    # ======================
    pass

# **********************
# Prepare Master PKL File
# **********************
pass

# **********************
# Collate Into motion_lib dataset
# **********************
pass

print("Done!")

# **********************
# Manifest.json File ?
# **********************
pass
