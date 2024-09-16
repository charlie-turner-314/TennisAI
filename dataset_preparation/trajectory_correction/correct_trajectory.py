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
) -> bool:
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
    frame = -1
    imputed = 0
    last_pose = None
    crop_info = None
    with open(cropped_json_file, "r") as f:
        crop_info = json.load(f)
    start_frame = crop_info[0]["frame"]
    ankles = []
    while True:
        ret, img = vid.read()
        if not ret:
            break
        frame += 1
        # check if crop_info contains the frame
        if not frame in [crop["frame"] for crop in crop_info]:
            continue
        # rel_frame is the index of the frame in the cropped video
        rel_frame = [crop["frame"] for crop in crop_info].index(frame)
        # get the ankles of this frame
        try:
            frame_pose = pose[rel_frame] 
        except:
            print(f"Frame {rel_frame} not found in pose file")
            print(f"Pose file has {len(pose)} frames")
            print(f"Crop file has {len(crop_info)} frames")

        assert frame_pose["frame"] == rel_frame

        left_ankle = (frame_pose.get("x15"), frame_pose.get("y15"))
        right_ankle = (frame_pose.get("x16"), frame_pose.get("y16"))
        # if they are 0, 0, then the pose is missing so set to None
        if left_ankle == (0, 0):
            left_ankle = (np.nan, np.nan)
        if right_ankle == (0, 0):
            right_ankle = (np.nan, np.nan)
        if left_ankle == (None, None):
            left_ankle = (np.nan, np.nan)
        if right_ankle == (None, None):
            right_ankle = (np.nan, np.nan)
        ankles.append((left_ankle, right_ankle))

    # now we have all of the ankles, interpolate any missing ones
    # there may be multiple missing in a row, so need to use linear interpolation both directions
    ankles = np.array(ankles)  # Assuming shape (N, 2, 2)
    print(ankles.shape)  # (187, 2, 2)

    for i in range(2):
        # Interpolate x and y separately for each ankle
        for j in range(2):
            ankle = np.array(ankles[:, j, i])
            # Find the indices of the missing values
            try:
                missing = np.isnan(ankle)
            except:
                print(ankle)
                raise ValueError("Ankle is not a numpy array")
            # Find the indices of the non-missing values
            not_missing = ~missing

            # Proceed if there are missing values and at least two non-missing values for interpolation
            if np.sum(not_missing) > 1 and np.sum(missing) > 0:
                # Perform interpolation
                interp_values = np.interp(
                    np.flatnonzero(missing),        # Indices of the missing values
                    np.flatnonzero(not_missing),    # Indices of the known values
                    ankle[not_missing]              # Values of the known data points
                )
                # Assign the interpolated values to the missing indices
                ankle[missing] = interp_values
            elif np.sum(missing) > len(ankle) - 2:
                return False
            
            # Assign the updated ankle back into the original array
            ankles[:, j, i] = ankle

    all_center_coords_court = []
    for rel_frame in range(len(ankles)):
        # correct the pose locations by accounting for cropped video dimensions
        crop = crop_info[rel_frame]
        left_ankle, right_ankle = ankles[rel_frame]

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
        if len(all_center_coords_court) > 0:
            i = 1 
            last_valid = all_center_coords_court[-i]
            while last_valid[0] is None:
                last_valid = all_center_coords_court[-i]
                i += 1
            diff = np.linalg.norm(np.array((court_x, court_y)) - np.array(last_valid))
            # NOTE: Magic Number
            # more than 2 meter in one frame, around 60 m/s is too fast. Must be an error
            if diff > i*2:
                # set the coords to np.nan
                court_x = court_y = np.nan
                imputed += 1
        all_center_coords_court.append((court_x, court_y))

    # linearly interpolate court coords with numpy
    all_center_coords_court = np.array(all_center_coords_court)
    num_coords = len(all_center_coords_court)
    all_center_coords_court = np.array(all_center_coords_court)
    print(all_center_coords_court.shape)
    missing = np.isnan(all_center_coords_court)
    if np.sum(~missing) > 1 and np.sum(missing) > 0:
        for i in range(2):
            all_center_coords_court[:, i] = np.interp(
                np.arange(num_coords),
                np.arange(num_coords)[~missing[:, i]],
                all_center_coords_court[~missing[:, i], i],
            )

    all_center_coords_pixels = []
    for frame in range(len(all_center_coords_court)):
        court_x, court_y = all_center_coords_court[frame]
        if court_x is None:
            all_center_coords_pixels.append((np.nan, np.nan))
            continue
        pixel = cv2.projectPoints(
            np.array([[court_x, court_y, 0]], dtype=np.float32),
            rvec,
            tvec,
            K,
            None,
        )[0][0][0]
        all_center_coords_pixels.append((pixel[0], pixel[1]))

    assert len(trans) == len(all_center_coords_court)

    for frame in range(len(all_center_coords_court)):
        court_x, court_y = all_center_coords_court[frame]
        assert court_x is not None and court_y is not None
        # update trans
        frame_trans = trans[frame]
        frame_trans[:2] = [court_x, court_y]
        trans[frame] = frame_trans

        # correct root orientation using rvec
        # TODO: Not sure how this works when the player has already been rotated in process_hybrik_data.py
        # R = cv2.Rodrigues(rvec)[0]  # rotation matrix
        # root_orient[frame] = R @ root_orient[frame]
        center = all_center_coords_pixels[frame]

        if save_video:
            # get the frame
            orig_frame = crop_info[frame]["frame"]
            vid.set(cv2.CAP_PROP_POS_FRAMES, orig_frame)
            ret, img = vid.read()
            # overlay the text onto the image
            text = f"({frame_trans[0]:.2}, {frame_trans[1]:.2}, {frame_trans[2]:.2})"
            img = cv2.putText(
                img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            # plot the center point
            img = cv2.circle(img, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
            img = cv2.resize(img, (width, height))
            new_vid.write(img)

    vid.release()
    if save_video:
        new_vid.release()

    # save the mesh back to the file
    mesh["trans"] = trans
    # need to check if any nans still
    # if more than 50% nans in any column, then return False - we can't use this clip
    for i in range(trans.shape[1]):
        if np.sum(np.isnan(trans[:, i])) > 0.5 * trans.shape[0]:
            return False

    # mesh["pose_aa"][:, :3] = root_orient
    mesh = {key: mesh}
    mesh_file_out = os.path.join(out_dir, os.path.basename(processed_mesh_file))
    with open(mesh_file_out, "wb") as f:
        joblib.dump(mesh, f)

    print(f"Imputed {imputed} frames out of {frame} frames")
    return True
