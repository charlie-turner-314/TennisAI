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
from Training.preprocess.preprocess_tennis import process_files, save_results
from dataset_preparation.trajectory_correction.correct_trajectory import (
    correct_hybrik_mesh,
)

SKIP_EXISTING = True
DEVICE = "cuda:0"


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
    help="Directory to save processed pkl dump.",
    dest="output_dir",
)

parser.add_argument(
    "--motion_lib_dir",
    type=str,
    default="Training/vid2player3d/data/motion_lib/tennis",
    help="Directory to output motion_lib dataset.",
    dest="mlib_dir",
)

args = parser.parse_args()

print("Preparing dataset using videos from: ", args.input_dir)
print("Outputs will be saved to: ", args.output_dir)
time.sleep(2)

all_videos = os.listdir(args.input_dir)
all_videos = [v for v in all_videos if v.endswith(".mp4")]
aux_dir = os.path.join(args.input_dir, "..", "aux")
num_processed = 0

for video_name in all_videos:
    video_path = os.path.join(args.input_dir, video_name)
    video_name = video_name.split(".")[0]
    print(COLORS.BLUE + f"Processing video: {video_path}" + COLORS.ENDC)
    # ======================
    # Crop To Player
    # ======================
    # player bounding box for each frame
    crop_check_file = os.path.join(aux_dir, "cropped", video_name + ".mp4")
    if SKIP_EXISTING and os.path.exists(crop_check_file):
        print(
            COLORS.GREEN
            + f"{crop_check_file} already exists... Skipping crop"
            + COLORS.ENDC
        )
        cropped_video_path = crop_check_file
    else:
        (bbox_w, bbox_h), positions = detect_player(filepath=video_path, player="fg")
        # Crop video to player bounding box
        success, cropped_video_path = crop_video(
            video_path, positions, bbox_w, bbox_h, os.path.join(aux_dir, "cropped")
        )
        if not success:
            print(f"Failed to crop video {video_path}.")
            continue
        print(
            COLORS.GREEN + f"Saved cropped video to: {cropped_video_path}" + COLORS.ENDC
        )
    # ======================
    # 2D Pose Estimation
    # ======================
    pose_check_file = os.path.join(aux_dir, "pose", video_name + ".json")
    if SKIP_EXISTING and os.path.exists(pose_check_file):
        print(
            COLORS.GREEN
            + f"{pose_check_file} already exists... Skipping pose estimation"
            + COLORS.ENDC
        )
    else:
        detect_pose(
            video=cropped_video_path,
            out_dir=os.path.join(aux_dir, "pose"),
            save_video=True,
            device=DEVICE,
        )
        print(
            COLORS.GREEN
            + f"Saved pose estimation to: {os.path.join(aux_dir, 'pose')}"
            + COLORS.ENDC
        )
    # ======================
    # 3D Mesh Estimation
    # ======================
    out_file = os.path.join(aux_dir, "cropped", video_name + ".mp4")
    out_file = os.path.abspath(out_file)
    out_dir = os.path.join(aux_dir, "pose_3d")
    out_dir = os.path.abspath(out_dir)

    mesh_check_file = os.path.join(out_dir, f"res_{video_name}.pk")
    if SKIP_EXISTING and os.path.exists(mesh_check_file):
        print(
            COLORS.GREEN
            + f"{mesh_check_file} already exists... Skipping 3D mesh estimation"
            + COLORS.ENDC
        )
    else:
        os.chdir("dataset_preparation/estimate_pose_3d/HybrIK")
        from dataset_preparation.estimate_pose_3d.estimate_pose import estimate_pose_3d

        estimate_pose_3d(
            video_name=out_file,
            out_dir=out_dir,
            save_video=False,
            device=DEVICE,
        )
        print(
            COLORS.GREEN
            + f"Saved 3D pose estimation to: {os.path.join(aux_dir, 'pose_3d')}"
            + COLORS.ENDC
        )
        os.chdir("../../..")
    # ======================
    # Tennis Court Line Detection
    # ======================
    # TODO: this takes a while -> perhaps a tag in the filename that says which long-form video it came from, then only do that once.
    # step 1: Convert the video to an avi with ffmpeg
    infile = video_path
    avi_file = os.path.join(aux_dir, "avi", "video.avi")

    # Each video should have {point}_{hit}.mp4
    # one lines file per point
    point = video_name.split("_")[0]
    lines_file = os.path.join(aux_dir, "lines", f"{point}.txt")

    if SKIP_EXISTING and os.path.exists(lines_file):
        print(
            COLORS.GREEN
            + f"{lines_file} already exists... Skipping court line detection"
            + COLORS.ENDC
        )
    else:
        os.makedirs(os.path.dirname(avi_file), exist_ok=True)
        command = f"ffmpeg -i {infile} -vcodec mjpeg -q:v 3 -acodec pcm_s16le {avi_file} -loglevel quiet -y"
        res = os.system(command)
        if res != 0:
            print(COLORS.RED + "Failed to convert video to AVI" + COLORS.ENDC)
            exit(1)
        # Run the c++ application to detect tennis court lines in the video
        os.makedirs(os.path.dirname(lines_file), exist_ok=True)
        command = f"dataset_preparation/court_detection/tennis-court-detection/build/bin/detect {avi_file} {lines_file}"
        res = os.system(command)
        if res != 0:
            print(COLORS.RED + "Failed to detect court lines in video." + COLORS.ENDC)
            exit(1)
    # ======================
    # Process HybrIK Data (Rotate to global frame)
    # ======================
    processed_check_file = os.path.join(
        aux_dir, "pose_3d", "processed", f"{video_name}.pkl"
    )
    hybrik_file = os.path.abspath(
        os.path.join(aux_dir, "pose_3d", f"res_{video_name}.pk")
    )
    output_file = os.path.abspath(
        os.path.join(aux_dir, "pose_3d", "processed", f"{video_name}.pkl")
    )
    if SKIP_EXISTING and os.path.exists(processed_check_file):
        print(
            COLORS.GREEN
            + f"{processed_check_file} already exists... Skipping Hybrik processing"
            + COLORS.ENDC
        )
        num_processed += 1
    else:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        os.chdir("Training/vid2player3d")
        process_hybrik(input_file=hybrik_file, output_file=output_file)
        os.chdir("../..")
    # ======================
    # Convert root position and orientation to court coordinates
    # ======================
    corrected_check_file = os.path.join(
        aux_dir, "pose", "corrected", f"{video_name}.pkl"
    )
    if SKIP_EXISTING and os.path.exists(corrected_check_file):
        print(
            COLORS.GREEN
            + f"{corrected_check_file} already exists... Skipping trajectory correction"
            + COLORS.ENDC
        )
    else:
        os.makedirs(os.path.join(aux_dir, "pose", "corrected"), exist_ok=True)
        correct_hybrik_mesh(
            processed_mesh_file=output_file,
            pose_file=os.path.join(aux_dir, "pose", f"{video_name}.json"),
            video_file=video_path,
            lines_file=lines_file,
            out_dir=os.path.join(aux_dir, "pose", "corrected"),
            cropped_json_file=os.path.join(aux_dir, "cropped", f"{video_name}.json"),
            save_video=True,
        )
    num_processed += 1

# **********************
# Prepare Master PKL File
# **********************
os.makedirs(args.output_dir, exist_ok=True)
results = process_files(os.path.join(aux_dir, "pose", "corrected"))
num_processed = len(results.keys())
save_results(results, os.path.join(args.output_dir, "processed_data.pkl"))
print(
    COLORS.GREEN + f"Saved {num_processed} motions to: {args.output_dir}" + COLORS.ENDC
)

# **********************
# Collate Into motion_lib dataset
# **********************
abs_output_path = os.path.abspath(args.output_dir)
abs_mlib_path = os.path.abspath(args.mlib_dir)
os.chdir("Training/vid2player3d")
command = f"""python uhc/utils/convert_amass_isaac.py \
            --amass_data {os.path.join(abs_output_path, "processed_data.pkl")} \
            --out_dir {abs_mlib_path} \
            --num_motion_libs {min(14, num_processed)}"""
res = os.system(command)
if res != 0:
    print(
        COLORS.RED
        + "Failed to convert processed data to motion_lib dataset."
        + COLORS.ENDC
    )
    exit(1)

print(COLORS.GREEN + f"Saved motion_lib data to: {args.mlib_dir}" + COLORS.ENDC)

print("Done!")

# **********************
# Manifest.json File ?
# **********************
pass
