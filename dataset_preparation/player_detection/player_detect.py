from ultralytics import YOLO
import numpy as np
import cv2
import os
import json

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

    writer = cv2.VideoWriter(
        "output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (video_width, video_height),
    )

    ret = True

    positions = {}  # Positions of each tracked person in each frame

    # generate random colors for the bounding boxes
    colors = np.random.randint(0, 255, (100, 3))

    frameId = 0
    actual_frames = 0
    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        actual_frames += 1
        results = model.track(frame, persist=True, verbose=False)
        img = frame.copy()
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
                    # Draw bounding box
                    [x1, y1, x2, y2] = b.xyxy[0].int().tolist()
                    color = tuple(colors[id % 100])
                    color = (int(color[0]), int(color[1]), int(color[2]))
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        img,
                        f"ID: {id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2,
                    )
        writer.write(img)
        frameId += 1

    cap.release()
    writer.release()

    print("Actual frames:", actual_frames)
    positions = {k: v[:actual_frames] for k, v in positions.items()}

    # save positions to a json file
    prefix = os.path.basename(filepath).split(".")[0]

    # the player we want will be the human who spends the most time in the bottom middle half of the frame
    # NOTE: This is a very simple heuristic, and WILL PROBABLY NOT WORK for anything other than broadcast video
    average_positions = {}
    variance_positions = {}
    for k in positions:
        # get average (where not all zeros)
        average_positions[k] = np.mean(
            positions[k][positions[k].sum(axis=1) != 0], axis=0
        )
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

    # make sure our person has a box for first frame
    while np.all(person[0] == 0):
        print(f"Person {ids[best_id]} doesn't have a bounding box in the first frame")
        best_id += 1
        person = positions[ids[best_id]]

    # now we have to go through each frame, and:
    # 1. See if there is a bounding box for our person
    # 2. If there is, great! If not, see if another person has a bounding box very close to the last sighting of our person
    # 3. If there is, 'switch' to that person, and continue
    # 4. If there isn't, remember we are missing this frame, and continue
    CLOSE = 0.1  # How close the bounding box has to be to the last sighting to be considered the same person (in proportion of the video height/width)
    print(
        "Our person is person",
        ids[best_id],
        "with average position",
        av_positions[best_id],
    )

    our_person = []
    missing_ids = []
    last_sighting = None  # center point (x, y) of the last sighting of our person

    num_frames = len(person)

    for i in range(num_frames):  # For each frame
        if np.all(person[i] == 0):
            # We don't have a bounding box for this person
            # Check if another person is close to the last sighting
            found = False
            for j in range(len(av_positions)):
                if np.all(positions[ids[j]][i] == 0):
                    continue  # this person doesn't have a box either
                candidate_position = positions[ids[j]][i]
                if (
                    last_sighting is not None
                    and np.linalg.norm(
                        np.array(candidate_position) - np.array(last_sighting)
                    )
                    < CLOSE
                ):
                    # This person is close enough to the last sighting
                    print(
                        "Last sighting:",
                        last_sighting,
                        "Candidate center:",
                        candidate_position,
                    )
                    print(f"Switched to person {ids[j]}")
                    person = positions[ids[j]]
                    our_person.append(person[i])
                    last_sighting = candidate_position
                    found = True
                    break
            if not found:
                # No person close enough to the last sighting
                our_person.append([0, 0, 0, 0])
                missing_ids.append(i)
        else:  # Just a usual frame tracking our person
            our_person.append(person[i])
            last_sighting = person[i]

    print(len(missing_ids), "frames missing")

    # if frames are missing at the end, just remove them
    removed_missing = False
    while np.all(our_person[-1] == 0):
        our_person.pop()
    if removed_missing:
        print("Removed missing end frames, new length: ", len(our_person))

    # make sure all elements are np arrays
    our_person = np.array(our_person)

    # interpolate other missing frames
    for i in missing_ids:
        # find the closest frame with a bounding box
        left = i - 1
        right = i + 1
        while np.all(our_person[left] == 0):
            left -= 1
        while np.all(our_person[right] == 0):
            right += 1
        # interpolate
        our_person[i] = (
            (our_person[left][0] + our_person[right][0]) / 2,
            (our_person[left][1] + our_person[right][1]) / 2,
            (our_person[left][2] + our_person[right][2]) / 2,
            (our_person[left][3] + our_person[right][3]) / 2,
        )

    # # find maximum height and width of the bounding box
    bbox_max_h = 0
    bbox_max_w = 0
    bbox_max_h = max([y2 - y1 for x1, y1, x2, y2 in our_person]) * video_height
    bbox_max_w = max([x2 - x1 for x1, y1, x2, y2 in our_person]) * video_width

    bbox_max_h = int(bbox_max_h * (1 + PADDING))
    bbox_max_w = int(bbox_max_w * (1 + PADDING))

    # denormalise person bounding boxes
    our_person = np.array(
        [
            (
                int(x1 * video_width),
                int(y1 * video_height),
                int(x2 * video_width),
                int(y2 * video_height),
            )
            for x1, y1, x2, y2 in our_person
        ]
    )

    return (bbox_max_w, bbox_max_h), our_person
