from ultralytics import YOLO
import numpy as np
import cv2

PERSON_CLASS = 0  # YOLO class for person
PADDING = 0.05  # Padding for the bounding box


def detect_player(filepath: str, player: str) -> "tuple[tuple[int, int], np.ndarray]":
    """
    player: str. One of ["fg", "bg"]
    """
    model = YOLO("yolov8s.pt")

    cap = cv2.VideoCapture(filepath)
    video_width, video_height, fps = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FPS)),
    )
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret = True

    positions = {}  # Positions of each tracked person in each frame

    frameId = 0
    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(frame, persist=True, verbose=False)
        for r in results:
            if r.boxes is None:
                break
            for b in r.boxes:
                # each b is a detection
                if b.cls == PERSON_CLASS and b.id is not None:
                    id = int(b.id[0])  # Tracking ID of the person
                    # Store the position of the person in the frame (xyxyn format)
                    if id not in positions:
                        positions[id] = np.zeros((num_frames, 4))  # x1, y1, x2, y2
                    positions[id][frameId] = b.xyxyn
        frameId += 1

    cap.release()

    # the player we want will be the human who spends the most time in the bottom middle half of the frame
    # NOTE: This is a very simple heuristic, and WILL PROBABLY NOT WORK for anything other than broadcast video
    average_positions = {}
    variance_positions = {}
    for k in positions:
        # get average (where not all zeros)
        average_positions[k] = np.mean(positions[k], axis=0)
        average_positions[k] = (
            (average_positions[k][0] + average_positions[k][2]) / 2,
            (average_positions[k][1] + average_positions[k][3]) / 2,
        )
    target_x = 0.5
    target_y = 0.75 if player == "fg" else 0.25

    # Sort by distance to target
    ids = list(average_positions.keys())
    av_positions = list(average_positions.values())
    ids, av_positions = zip(
        *sorted(
            zip(ids, av_positions),
            key=lambda x: (x[1][0] - target_x) ** 2 + (x[1][1] - target_y) ** 2,
        )
    )

    best_id = 0
    person = positions[
        ids[best_id]
    ]  # bounding box in each frame for the person we want

    # we need at least 90% of the frames to have a bounding box for the person or we skip the video
    # NOTE: Magic Number
    while np.sum(np.sum(person, axis=1) != 0) < 0.9 * num_frames:
        num_frames_appear = np.sum(
            np.sum(person, axis=1) != 0
        )  # number of frames the person appears in
        print(
            f"person {int(ids[best_id])} does not appear in enough frames ({num_frames_appear} / {num_frames}), trying next person..."
        )

        if best_id + 1 < len(ids):
            print("No more people in the video")
            break
        best_id += 1
        person = positions[ids[best_id]]

    # # find maximum height and width of the bounding box
    bbox_max_h = 0
    bbox_max_w = 0
    bbox_max_h = max([y2 - y1 for x1, y1, x2, y2 in person]) * video_height
    bbox_max_w = max([x2 - x1 for x1, y1, x2, y2 in person]) * video_width

    bbox_max_h = int(bbox_max_h * (1 + PADDING))
    bbox_max_w = int(bbox_max_w * (1 + PADDING))

    # denormalise person bounding boxes
    person = np.array(
        [
            (
                int(x1 * video_width),
                int(y1 * video_height),
                int(x2 * video_width),
                int(y2 * video_height),
            )
            for x1, y1, x2, y2 in person
        ]
    )

    return (bbox_max_w, bbox_max_h), person
