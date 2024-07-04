import os
import cv2
import numpy as np
import json
from ultralytics import YOLO

MODEL = "yolov8x-pose"


def detect_pose(video, out_dir, save_video=False, device="cpu"):
    os.makedirs(out_dir, exist_ok=True)
    video_name = os.path.basename(video)
    output_path = os.path.join(out_dir, video_name.replace(".mp4", ".json"))

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    model = YOLO(MODEL)
    results = model(video, device=device, stream=True, verbose=False)
    keypoints_data = []
    frame_id = 0

    if save_video:
        writer = cv2.VideoWriter(
            os.path.join(out_dir, f"{os.path.basename(video)}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (video_width, video_height),
        )

    for r in results:
        annot_frame = r.plot(boxes=False, labels=False, probs=False)
        keypoints = r.keypoints
        kp_data_frame = {"frame": frame_id}
        for k in keypoints:
            if len(k.data) != 1:
                print("Multiple sets of keypoints detected in a single frame")
            for i, keypoint in enumerate(k.data[0]):
                x, y, conf = keypoint
                kp_data_frame[f"x{i}"] = int(x)
                kp_data_frame[f"y{i}"] = int(y)
        keypoints_data.append(kp_data_frame)
        if save_video:
            writer.write(annot_frame)
        frame_id += 1

    # Save json
    with open(output_path, "w") as f:
        json.dump(keypoints_data, f)

    if save_video:
        writer.release()