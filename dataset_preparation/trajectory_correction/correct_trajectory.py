import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import joblib
import json


def get_intrinsic_props(video_file):
    """
    Get the intrinsic camera properties from a video file. Here we use:
        - F_x = F_y = max(vid_width, vid_height)
        - C_x = vid_width / 2
        - C_y = vid_height / 2
    Returns:
        - (F_x, F_y, C_x, C_y)
    """
    vid = cv2.VideoCapture(video_file)
    if not vid.isOpened():
        # Check exists
        if not os.path.exists(video_file):
            raise FileNotFoundError(f"Could not find video file {video_file}")
        raise ValueError(f"Could not open video file {video_file}")
    vid_width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    vid_height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    F_x = F_y = max(vid_width, vid_height)
    C_x = vid_width / 2
    C_y = vid_height / 2
    vid.release()
    return F_x, F_y, C_x, C_y


def solve_PNP(video_file, txt_file):
    """
    Solve the PNP problem using the points defined in the txt file.
    Returns:
        - camera matrix
    """
    F_x, F_y, C_x, C_y = get_intrinsic_props(video_file)
    # Read the txt file
    with open(txt_file, "r") as f:
        lines = f.readlines()
    points = []
    for line in lines:
        x, y = line.split(";")
        points.append((float(x), float(y)))
    # define the 3D points
    """ 
    x ->
    y goes up
    z comes out of the screen (towards head from ground)
    |-|-------------|-|
    | |             | |
    | |             | |
    | |-------------| |
    | |      |      | |
    | |      |      | |
   -|-|------O------|-|-
    | |      |      | |
    | |      |      | |
    | |-------------| |
    | |             | |
    | |             | |
    |-|-------------|-|

            ^
            |
         |camera|
    """
    # Define lengths in meters from origin to key points
    BASELINE = 11.89
    SIDELINE = 5.49
    SERVICE_LINE = 6.4
    SINGLES_LINE = 4.11
    obj_points = [
        (-SIDELINE, BASELINE),  # 1. top left - +
        (-SIDELINE, -BASELINE),  # 2. bottom left - -
        (SIDELINE, -BASELINE),  # 3. bottom right + -
        (SIDELINE, BASELINE),  # 4. top right + +
        (-SINGLES_LINE, BASELINE),  # 5. Top left of singles line
        (-SINGLES_LINE, -BASELINE),  # 6. Bottom left of singles line
        (SINGLES_LINE, -BASELINE),  # 7. Bottom right of singles line
        (SINGLES_LINE, BASELINE),  # 8. Top right of singles line
        (-SINGLES_LINE, SERVICE_LINE),  # 9. Top left of service line
        (SINGLES_LINE, SERVICE_LINE),  # 10. Top right of service line
        (-SINGLES_LINE, -SERVICE_LINE),  # 11. Bottom left service line
        (SINGLES_LINE, -SERVICE_LINE),  # 12. Bottom right service line
        (0, SERVICE_LINE),  # 13. Top middle service line
        (0, -SERVICE_LINE),  # 14. bottom middle service line
        (-SIDELINE, 0),  # 15. Left net line
        (SIDELINE, 0),  # 16. Right net line
    ]
    # add a column of zeros to the object points (z-coordinate)
    obj_points = np.array(obj_points, dtype=np.float32)
    obj_points = np.hstack(
        (obj_points, np.zeros((obj_points.shape[0], 1), dtype=np.float32))
    )

    # solve the PNP problem
    obj_points = np.array(obj_points, dtype=np.float32)
    points = np.array(points, dtype=np.float32)
    # Fill the intrinsic camera matrix (initial guess)
    camera_matrix = np.zeros((3, 3), dtype=np.float32)
    camera_matrix[0, 0] = F_x
    camera_matrix[1, 1] = F_y
    camera_matrix[0, 2] = C_x
    camera_matrix[1, 2] = C_y
    camera_matrix[2, 2] = 1
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    image_size = (1920, 1080)
    _, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        [obj_points],
        [points],
        image_size,
        camera_matrix,
        dist_coeffs,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS,
    )
    rvec = rvecs[0]
    tvec = tvecs[0]

    return cameraMatrix, rvec, tvec, lines


def pixel_to_court_coords(
    x, y, clip_file=None, lines_file=None, K=None, rvec=None, tvec=None, lines=None
):
    # TODO: use filename for court txt as well!
    if clip_file is not None and lines_file is not None:
        K, rvec, tvec, lines = solve_PNP(clip_file, lines_file)
    elif K is None or rvec is None or tvec is None:
        raise ValueError("Must provide either filenames or K, rvec, tvec")
    K_inv = np.linalg.inv(K)
    R = cv2.Rodrigues(rvec)[0]  # rotation matrix
    t = tvec.reshape(3)

    # Convert the pixel to camera coordinates (ray in camera coordinates)
    pixel = np.array([x, y, 1])
    camera_coords = K_inv @ pixel  # This is essentially a ray in camera coordinates

    # Transform the ray from camera coordinates to world coordinates
    # We need to apply the inverse of the rotation matrix R and translation vector t

    # Convert to homogeneous coordinates in camera frame
    camera_coords_homogeneous = np.append(camera_coords, [1])

    # The ray in world coordinates
    ray_world = R.T @ (camera_coords - t)

    # The plane equation Z = 0 (in world coordinates)
    # The parametric equation of the ray: P = O + t * D
    # Where P is the point on the ray, O is the origin (camera position), and D is the direction (ray)
    # We need to find t where P_z = 0 (Z coordinate is 0)

    # Camera origin in world coordinates
    camera_origin_world = -R.T @ t

    # Direction of the ray in world coordinates
    ray_direction_world = ray_world - camera_origin_world

    # Calculate the parameter t for the plane Z=0
    t_intersect = -camera_origin_world[2] / ray_direction_world[2]

    # The 3D point of intersection on the plane Z=0
    intersection_world = camera_origin_world + t_intersect * ray_direction_world

    return intersection_world[:2]


def correct_hybrik_mesh(
    processed_mesh_file: str,
    pose_file: str,
    video_file: str,
    lines_file: str,
    out_dir: str,
    cropped_json_file: str,
    save_video: bool = False,
):
    filename = os.path.basename(video_file).split(".")[0]
    # use first frame
    frame = 0
    with open(processed_mesh_file, "rb") as f:
        mesh = joblib.load(f)
        key = list(mesh.keys())[0]
        mesh = mesh[key]
    # Mesh has: "filename" : {root_orient, trans, pose_aa, beta, joints2d}

    # load the 2D pose from data/pose_estimations
    with open(pose_file, "r") as f:
        pose = json.load(f)

    trans = mesh["trans"]
    root_orient = mesh["pose_aa"][:, :3]
    # for each frame, compute the court coordinates of the ankle center -> change the x and y of trans to that
    # Overlay the trans coordinates on each frame of the video and save as a new video

    K, rvec, tvec, lines = solve_PNP(video_file=video_file, txt_file=lines_file)

    vid = cv2.VideoCapture(video_file)
    if not vid.isOpened():
        raise ValueError("Could not open video file")
    # Get the video properties
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if save_video:
        new_vid = cv2.VideoWriter(
            os.path.join(out_dir, f"translated_{os.path.basename(video_file)}"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
    frame = 0
    last_coords = []
    imputed = 0
    while True:
        ret, img = vid.read()
        if not ret:
            break
        # Get the trans of this frame
        frame_trans = trans[frame]
        # get the ankles of this frame
        frame_pose = pose[frame]

        if frame_pose["frame"] != frame:
            raise ValueError("Frame number does not match")

        left_ankle = (frame_pose.get("x15"), frame_pose.get("y15"))
        right_ankle = (frame_pose.get("x16"), frame_pose.get("y16"))
        if left_ankle is None:
            left_ankle = right_ankle
        if right_ankle is None:
            right_ankle = left_ankle

        if left_ankle is None or right_ankle is None:
            raise ValueError("Ankle location not found")

        # correct the pose locations by accounting for cropped video dimensions
        with open(cropped_json_file, "r") as f:
            crop = json.load(f)
        crop = crop[frame]
        if crop["frame"] != frame:
            raise ValueError("Frame number does not match")
        left_ankle = (left_ankle[0] + crop["x1"], left_ankle[1] + crop["y1"])
        right_ankle = (right_ankle[0] + crop["x1"], right_ankle[1] + crop["y1"])

        center = (
            (left_ankle[0] + right_ankle[0]) / 2,
            (left_ankle[1] + right_ankle[1]) / 2,
        )
        court_x, court_y = pixel_to_court_coords(
            center[0], center[1], K=K, rvec=rvec, tvec=tvec, lines=lines
        )

        # Sanity check -> how much has the player moved?
        if len(last_coords) > 0:
            diff = np.linalg.norm(
                np.array((court_x, court_y)) - np.array(last_coords[-1])
            )
            # NOTE: Magic Number
            if diff > 1:  # more than 1 meter
                # TODO: GLAMR - Too hard to work out
                # For now. Use the moving av of up to 5 frames velocity to impute this frame
                ma_len = 5 if len(last_coords) > 5 else len(last_coords)
                vel = np.mean(np.diff(last_coords[-ma_len:], axis=0), axis=0)
                # Impute the position
                court_x, court_y = last_coords[-1] + vel
                imputed += 1

        last_coords.append((court_x, court_y))

        if court_x is None or court_y is None:
            raise ValueError("Court coordinates not found for frame", frame)
        # update trans
        frame_trans[:2] = [court_x, court_y]
        trans[frame] = frame_trans

        # correct root orientation using rvec
        # TODO: Not sure how this works when the player has already been rotated in process_hybrik_data.py
        # R = cv2.Rodrigues(rvec)[0]  # rotation matrix
        # root_orient[frame] = R @ root_orient[frame]

        if save_video:
            # overlay the text onto the image
            text = f"({frame_trans[0]:.2}, {frame_trans[1]:.2}, {frame_trans[2]:.2})"
            img = cv2.putText(
                img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            # plot the center point
            img = cv2.circle(img, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
            img = cv2.resize(img, (width, height))
            new_vid.write(img)
        frame += 1
    vid.release()
    if save_video:
        new_vid.release()

    # save the mesh back to the file
    mesh["trans"] = trans
    # mesh["pose_aa"][:, :3] = root_orient
    mesh = {key: mesh}
    mesh_file_out = os.path.join(out_dir, os.path.basename(processed_mesh_file))
    with open(mesh_file_out, "wb") as f:
        joblib.dump(mesh, f)

    print(f"Imputed {imputed} frames out of {frame} frames")
