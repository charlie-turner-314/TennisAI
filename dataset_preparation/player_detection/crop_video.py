import cv2
import numpy as np
import os
import json


def crop_video(filename, position, bbox_w, bbox_h, out_dir) -> "tuple[bool, str]":
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print(f"Error opening video {filename}")
        return False
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(out_dir, exist_ok=True)
    out_video = os.path.join(out_dir, os.path.basename(filename))
    out_json = out_video.replace(".mp4", ".json")  # Video writer for avi
    out_cap_cropped = cv2.VideoWriter(
        out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (bbox_w, bbox_h)
    )
    out_cap_annotations = cv2.VideoWriter(
        out_video.replace(".mp4", "_annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (video_width, video_height),
    )
    # JSON writer
    json_data = []
    ret = True
    idx = 0
    while ret:
        ret, frame = cap.read()
        if frame is None:
            break

        if idx >= len(position):
            break
        x1, y1, x2, y2 = position[idx]  # Original (detected) Bounding box
        if (
            np.sum([x1, y1, x2, y2]) == 0
        ):  # TODO: Currently skipping if no person detected in frame, could interpolate instead
            print(f"Person not detected in frame {idx}")
            idx += 1
            continue

        # Get the center of the bounding box
        center_x, center_y = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Get the new bounding box
        x1_crop = max(0, center_x - bbox_w // 2)
        x2_crop = min(frame.shape[1], center_x + bbox_w // 2)
        y1_crop = max(0, center_y - bbox_h // 2)
        y2_crop = min(frame.shape[0], center_y + bbox_h // 2)

        # Correct for edge cases (pun intended)
        if x1_crop == 0:
            x2_crop = bbox_w
        if x2_crop == frame.shape[1]:
            x1_crop = frame.shape[1] - bbox_w
        if y1_crop == 0:
            y2_crop = bbox_h
        if y2_crop == frame.shape[0]:
            y1_crop = frame.shape[0] - bbox_h

        # Save the bounding box of the cropped video (video coords)
        json_data.append(
            {
                "frame": idx,
                "x1": int(x1_crop),
                "y1": int(y1_crop),
                "x2": int(x2_crop),
                "y2": int(y2_crop),
            }
        )

        cropped = frame[y1_crop:y2_crop, x1_crop:x2_crop]

        cropped = cv2.resize(
            cropped, (bbox_w, bbox_h)
        )  # Have to resize so video saves for some reason

        out_cap_cropped.write(cropped)

        # Draw the bounding box on the original frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        out_cap_annotations.write(frame)

        idx += 1

    with open(out_json, "w") as f:
        json.dump(json_data, f)

    cap.release()
    out_cap_cropped.release()
    out_cap_annotations.release()
    cv2.destroyAllWindows()
    return True, out_video
