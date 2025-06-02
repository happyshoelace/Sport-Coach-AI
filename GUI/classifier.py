import mediapipe as mp
import cv2
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import datetime
from poseCorrection import poseCorrection

# 1) Load your YOLO model once
yolo = YOLO('yolov8n.pt')  # or 'yolov5n.pt'


def isolate_largest_person(image: np.ndarray) -> np.ndarray:
    """
    Runs YOLO on `image`, finds all 'person' boxes,
    picks the one with the largest area, zeros out everything else,
    and returns an image of exactly the same size.
    """
    # Run inference
    results = yolo(image)[0]  # first frame of batch

    # Filter to only class 'person' ( class 0)
    persons = [box for box in results.boxes if int(box.cls) == 0]
    if not persons:
        return image  # no person found, fallback to original

    # Pick the box with largest area
    areas = [(b.xyxy[0][2].cpu().numpy() - b.xyxy[0][0].cpu().numpy()) * (b.xyxy[0][3].cpu().numpy() - b.xyxy[0][1].cpu().numpy())
             for b in persons]
    best = persons[int(np.argmax(areas))]
    x1, y1, x2, y2 = map(int, best.xyxy[0].cpu().tolist())

    # Create a mask that’s white inside the box, black outside
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    # Apply mask: keep the person, blackout the rest
    isolated = cv2.bitwise_and(image, mask)
    return isolated

def save_json(input_base_path, file_name, dominant_hand, output_base_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False)

    print(f"Processing video: {file_name} with dominant hand: {dominant_hand}")
    print(f"Input path: {input_base_path}, Output path: {output_base_path}")
    # --- CONFIGURE THESE ---
    video_path = os.path.join(input_base_path, file_name.split('/')[-1])
    output_folder = output_base_path
    output_file = f"{file_name.split('/')[-1].split('.')[0]}.json"
    print(f"Output file will be saved as: {output_file}")
    # -----------------------

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_data = []
    frame_idx = 0

    # --- VideoWriter setup ---
    fourcc = cv2.VideoWriter_fourcc(*'VP80')  # VP8 codec for webm
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video_path = os.path.join(output_folder, f"{file_name}_annotated.webm")  # Save as webm
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    # -------------------------

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = isolate_largest_person(frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)

        keypoints = []
        if pose_results.pose_landmarks:
            for lm in pose_results.pose_landmarks.landmark:
                keypoints.append({
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    'visibility': lm.visibility
                })

            # Optionally, draw pose landmarks on the frame for visualization
            mp.solutions.drawing_utils.draw_landmarks(
                frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        frame_data.append({
            'frame': frame_idx,
            'keypoints': keypoints,
            'dominantHand': dominant_hand
        })

        # --- Write processed frame to output video ---
        out.write(frame)
        # --------------------------------------------

        frame_idx += 1

    cap.release()
    out.release()  # Don't forget to release the VideoWriter!
    pose.close()

    output_path = os.path.join(output_folder, output_file)
    with open(output_path, 'w') as f:
        json.dump(frame_data, f, indent=2)

    with open(output_path + "_Raw.json", 'w') as f:
        json.dump(frame_data, f, indent=2)

    # input("...")

    print(f"Processing complete. JSON saved to: {output_path}")
    print(f"Processed video saved to: {output_video_path}")


import json

REDUNDANT_IDX = set(range(1, 11))  # Indices of facial keypoints to remove

def remove_facial_keypoints_single_json(json_path):
    """
    Loads a single JSON file containing frames with pose keypoints,
    removes facial keypoints (indices 1 to 10) from each frame's keypoints if length 33,
    and overwrites the same JSON file with the processed data.
    
    Args:
        json_path (str): Path to the JSON file to process.
    """
    with open(json_path, 'r') as f:
        frames = json.load(f)

    processed_frames = []
    for frame in frames:
        kp = frame.get("keypoints", [])
        if isinstance(kp, list) and len(kp) == 33:
            pruned_kp = [pt for i, pt in enumerate(kp) if i not in REDUNDANT_IDX]
            frame["keypoints"] = pruned_kp
        # else keep keypoints as-is
        processed_frames.append(frame)

    with open(json_path, 'w') as f:
        json.dump(processed_frames, f, indent=2)

    print(f"Processed {json_path}: kept {len(processed_frames)} frames")
ORIG_KEYS = list(range(33))
REDUNDANT = set(range(1, 11))
PRUNED_MAP = [k for k in ORIG_KEYS if k not in REDUNDANT]  # length 23

LH_IP_IDX = PRUNED_MAP.index(23)  # LEFT_HIP
RH_IP_IDX = PRUNED_MAP.index(24)  # RIGHT_HIP
LS_H_IDX  = PRUNED_MAP.index(11)  # LEFT_SHOULDER
RS_H_IDX  = PRUNED_MAP.index(12)  # RIGHT_SHOULDER

def normalize_single_json(json_path):
    with open(json_path, 'r') as f:
        frames = json.load(f)

    for frame in frames:
        kp = frame.get("keypoints", [])
        if not isinstance(kp, list) or len(kp) != len(PRUNED_MAP):
            continue

        arr = np.array([[pt["x"], pt["y"], pt["z"]] for pt in kp], dtype=float)

        # 1) center at hip midpoint
        center = 0.5 * (arr[LH_IP_IDX] + arr[RH_IP_IDX])

        # 2) scale by shoulder width (avoid divide by zero)
        raw_dist = np.linalg.norm(arr[LS_H_IDX] - arr[RS_H_IDX])
        scale = raw_dist if raw_dist > 0 else 1.0

        # 3) normalize
        arr = (arr - center) / scale

        # write back normalized coords
        for i, (x, y, z) in enumerate(arr):
            frame["keypoints"][i]["x"] = float(x)
            frame["keypoints"][i]["y"] = float(y)
            frame["keypoints"][i]["z"] = float(z)

    with open(json_path, 'w') as f:
        json.dump(frames, f, indent=2)

    print(f"Normalized {json_path} ({len(frames)} frames)")

import json
import numpy as np

THRESHOLD  = 10              # prune gaps longer than 10 frames
NUM_KP     = 23              # after pruning redundant points
DIM_PER_KP = 3               # x, y, z

def process_sequence(frames):
    """
    Given a list of frames (each a dict with frame["keypoints"]),
    returns (new_frames, coords_filled) where:
      - new_frames may be shorter if we pruned a long initial gap,
      - coords_filled is a (T', NUM_KP, 3) numpy array with no NaNs.
    """
    coords = []
    for f in frames:
        kp = f.get("keypoints", [])
        if isinstance(kp, list) and len(kp) == NUM_KP:
            arr = np.array([[pt["x"], pt["y"], pt["z"]] for pt in kp], dtype=float)
        else:
            arr = np.full((NUM_KP, DIM_PER_KP), np.nan, dtype=float)
        coords.append(arr)
    data = np.stack(coords, axis=0)  # shape (T, NUM_KP, 3)

    valid = ~np.isnan(data[:, 0, 0])
    if not valid.any():
        # no valid detections → nothing to interpolate
        return frames, None

    idx_valid = np.where(valid)[0]

    for a, b in zip(idx_valid, idx_valid[1:]):
        if b - a > THRESHOLD:
            # prune frames up to b (inclusive gap)
            return process_sequence(frames[b:])

    first = idx_valid[0]
    data[:first] = data[first]

    last = idx_valid[-1]
    data[last+1:] = data[last]

    for a, b in zip(idx_valid, idx_valid[1:]):
        L = b - a
        if 1 < L <= THRESHOLD:
            for k in range(a+1, b):
                alpha = (k - a) / float(L)
                data[k] = (1 - alpha) * data[a] + alpha * data[b]

    return frames, data


def process_single_json_sequence(json_path):
    """
    Load JSON frames from file, run process_sequence to fill gaps,
    update frames with interpolated coords, save back JSON.

    Args:
        json_path (str): path to JSON file
    """
    with open(json_path, 'r') as f:
        frames = json.load(f)

    frames, coords_filled = process_sequence(frames)

    if coords_filled is None:
        print(f"No valid frames found in {json_path}. No changes made.")
        return

    # Update keypoints in frames with coords_filled (which has no NaNs)
    for i, frame in enumerate(frames):
        kp = frame.get("keypoints", [])
        if isinstance(kp, list) and len(kp) == NUM_KP:
            for j in range(NUM_KP):
                x, y, z = coords_filled[i, j]
                frame["keypoints"][j]["x"] = float(x)
                frame["keypoints"][j]["y"] = float(y)
                frame["keypoints"][j]["z"] = float(z)

    # Save updated frames back to JSON
    with open(json_path, 'w') as f:
        json.dump(frames, f, indent=2)

    print(f"Processed and filled gaps in {json_path} ({len(frames)} frames)")

def fill_and_prune_single_json(json_path):
    """
    Processes one JSON file to fill missing keypoints and prune long empty gaps.
    Saves the result back to the JSON file.
    """
    with open(json_path, 'r') as f:
        frames = json.load(f)

    new_frames, filled_data = process_sequence(frames)

    if filled_data is not None:
        out_frames = []
        for orig_frame, arr in zip(new_frames, filled_data):
            old_kp = orig_frame.get("keypoints", [])
            new_kp = []
            for j, (x, y, z) in enumerate(arr):
                vis = (old_kp[j]["visibility"]
                       if isinstance(old_kp, list) and len(old_kp) == NUM_KP
                       else 0.0)
                new_kp.append({
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "visibility": float(vis)
                })
            orig_frame["keypoints"] = new_kp
            out_frames.append(orig_frame)
    else:
        out_frames = frames

    with open(json_path, 'w') as f:
        json.dump(out_frames, f, indent=2)

    print(f"Processed {json_path}: {len(frames)} → {len(out_frames)} frames")

import numpy as np

def compute_frame_angles(keypoints, file_path):
    """
    Compute a comprehensive, hand-agnostic set of joint angles from 23 keypoints.
    Returns: dict of angle_name → float (degrees)
    """
    if not isinstance(keypoints, list) or len(keypoints) != 23:
        print(f"Missing or malformed keypoints → {file_path}")
        return {}

    arr = np.array([[kp["x"], kp["y"], kp["z"]] for kp in keypoints], dtype=float)

    hip_mid = 0.5 * (arr[13] + arr[14])     # LEFT_HIP=13, RIGHT_HIP=14
    shoulder_mid = 0.5 * (arr[1] + arr[2])  # LEFT_SHOULDER=1, RIGHT_SHOULDER=2

    def angle(A, B, C):
        BA = A - B
        BC = C - B
        cosang = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

    def lateral_rotation(A, B, C):
        # Projects onto X-Z plane before angle calculation
        BA = (A - B) * np.array([1, 0, 1])
        BC = (C - B) * np.array([1, 0, 1])
        cosang = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

    # Leg joints
    left_knee  = angle(arr[13], arr[15], arr[17])  # LEFT_HIP → KNEE → ANKLE
    right_knee = angle(arr[14], arr[16], arr[18])  # RIGHT_HIP → KNEE → ANKLE

    left_ankle_dorsi  = angle(arr[15], arr[17], arr[21])  # Knee → Ankle → FootIndex
    right_ankle_dorsi = angle(arr[16], arr[18], arr[22])

    left_ankle_lat  = lateral_rotation(arr[15], arr[17], arr[21])
    right_ankle_lat = lateral_rotation(arr[16], arr[18], arr[22])

    # Arm joints
    left_elbow_angle  = angle(arr[1], arr[3], arr[5])   # SHOULDER → ELBOW → WRIST
    right_elbow_angle = angle(arr[2], arr[4], arr[6])

    left_arm_elevation  = angle(hip_mid, arr[1], arr[3])  # HIP MID → SHOULDER → ELBOW
    right_arm_elevation = angle(hip_mid, arr[2], arr[4])

    # Hip abduction
    left_hip_abd  = angle(shoulder_mid, arr[13], arr[15])
    right_hip_abd = angle(shoulder_mid, arr[14], arr[16])

    # Torso
    vert_pt = hip_mid + np.array([0.0, 1.0, 0.0])
    torso_lean = angle(vert_pt, hip_mid, shoulder_mid)

    inter_arm_angle = angle(arr[3], arr[0], arr[4])  # LEFT_ELBOW → NOSE → RIGHT_ELBOW

    angles = {
        "left_knee_flexion":             left_knee,
        "right_knee_flexion":            right_knee,
        "left_ankle_dorsiflexion":       left_ankle_dorsi,
        "right_ankle_dorsiflexion":      right_ankle_dorsi,
        "left_ankle_lateral_rotation":   left_ankle_lat,
        "right_ankle_lateral_rotation":  right_ankle_lat,
        "left_elbow_angle":              left_elbow_angle,
        "right_elbow_angle":             right_elbow_angle,
        "left_arm_elevation":            left_arm_elevation,
        "right_arm_elevation":           right_arm_elevation,
        "left_hip_abduction":            left_hip_abd,
        "right_hip_abduction":           right_hip_abd,
        "torso_lean":                    torso_lean,
        "inter_arm_angle":               inter_arm_angle,
    }

    return angles

def process_single_json_with_angles_in_place(input_json_path: str):
    """
    Loads one JSON, computes angles for each frame, and overwrites the same file.

    Args:
        input_json_path (str): Full path to the input JSON file.
    """
    # 1a) Load original frames
    with open(input_json_path, 'r') as f:
        frames = json.load(f)

    # 1b) Compute and attach angles per frame
    for frame in frames:
        frame["angles"] = compute_frame_angles(frame.get("keypoints", []), input_json_path)

    # 1c) Overwrite the original file with enriched frames
    with open(input_json_path, 'w') as f:
        json.dump(frames, f, indent=2)

    print(f"Updated JSON in place: {input_json_path}")

def extract_joint_windows(
    json_file: str,
    window_size: int = 24,
    step: int = 1,
    pad_short: bool = True
) -> np.ndarray:
    """
    Loads one JSON file of frames (with 'keypoints' and precomputed 'angles'),
    pads as needed, then slides a window of length `window_size` with stride `step`,
    returning an array of flattened [coords + angles].

    Now reads frame["angles"] instead of recomputing them.
    """
    # 1) Load frames
    with open(json_file, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        frames = data
    elif isinstance(data, dict) and 'keypoints' in data:
        frames = data['keypoints']
    else:
        # Handle unexpected data structure, perhaps skip this file or raise an error
        print(f"Skipping {json_file}: Unexpected data structure.")
        return np.empty((0,))

    num_frames = len(frames)
    if num_frames == 0:
        return np.empty((0,))

    # 2) Pad short sequences
    if pad_short and num_frames < window_size:
        last = frames[-1]
        frames = frames + [last] * (window_size - num_frames)
        num_frames = window_size

    # 3) Check keypoint consistency
    kp_lens = [len(f.get('keypoints', [])) for f in frames]
    nonzero = set(kp_lens) - {0}
    if len(nonzero) > 1:
        raise ValueError(f"Inconsistent keypoint counts: {set(kp_lens)}")
    num_kp = kp_lens[0] if kp_lens[0] > 0 else 0

    # 4) Discover angle names from the first frame that has them
    first_with_angles = next((f for f in frames if isinstance(f.get('angles'), dict)), None)
    angle_names = list(first_with_angles["angles"].keys()) if first_with_angles else []
    num_angles = len(angle_names)

    windows = []
    for start in range(0, num_frames - window_size + 1, step):
        block = frames[start:start + window_size]

        # a) coords: window_size x num_kp x 3 → flatten
        coords = []
        for frm in block:
            kp = frm.get('keypoints', [])
            if isinstance(kp, list) and len(kp) == num_kp:
                coords.append([[pt['x'], pt['y'], pt['z']] for pt in kp])
            else:
                coords.append([[0.0, 0.0, 0.0]] * num_kp)
        coords_flat = np.array(coords, dtype=float).reshape(-1)

        # b) angles: window_size x num_angles → flatten
        angles = []
        for frm in block:
            ang = frm.get('angles', {})
            if isinstance(ang, dict):
                angles.append([ang.get(name, 0.0) for name in angle_names])
            else:
                angles.append([0.0] * num_angles)
        angles_flat = np.array(angles, dtype=float).reshape(-1)

        # c) concatenate coords + angles
        windows.append(np.concatenate([coords_flat, angles_flat], axis=0))

    return np.stack(windows, axis=0)


from tensorflow.keras.models import load_model

def predict_windows_from_json(
    json_file_path: str,
    model_path: str = "model.keras",
    window_size: int = 24,
    step: int = 1
):
    """
    Loads the trained model and performs prediction over windows extracted
    from a single JSON file. Prints softmax outputs for each window.

    Args:
        json_file_path (str): Path to the input JSON with keypoints + angles.
        model_path (str): Path to the saved Keras model.
        window_size (int): Number of frames per window.
        step (int): Step size for sliding window.

    Returns:
        np.ndarray: Model softmax outputs, shape (num_windows, 4)
    """
    # 1) Load the trained model
    model = load_model(model_path)

    # 2) Extract windows from JSON
    windows = extract_joint_windows(json_file_path, window_size=window_size, step=step)

    if windows.size == 0:
        print("No windows extracted. Check JSON content.")
        return np.empty((0,))

    # 3) Reshape for LSTM input: (num_windows, window_size, features_per_frame)
    features_per_frame = windows.shape[1] // window_size
    X = windows.reshape((-1, window_size, features_per_frame))

    # 4) Run predictions
    predictions = model.predict(X)

    # 5) Print softmax output for each window
    for i, probs in enumerate(predictions):
        print(f"Window {i:02d}: {probs} → Predicted class: {np.argmax(probs)}")

    return predictions

def get_prediction_probability_and_index(probs):
    """
    probs: np.ndarray of shape (n_frames, n_classes)
    Returns: int, index of the most likely class
    """
    avg_probs = np.mean(probs, axis=0)  # average over frames
    result = int(np.argmax(avg_probs))
    # FIRST is prob eg 90% SECOND is index eg 2
    return float(np.max(avg_probs)), result


def process_video(input_path, output_path, pose_path, classification, sample_path):

        
    with open(pose_path, 'r') as f:
        pose_data = json.load(f)
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'VP80')  # 'XVID' or 'mp4v' for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def draw_point(num):
        cv2.ellipse(frame, (int(keypoints[num]['x'] * width), int(keypoints[num]['y'] * height)), (5, 5), 0, 0, 360, (220, 220, 220), -1)

    # order is first second third eg shoulder elbow wrist
    def draw_arc(num1, num2, num3, col):
        a, b, c = (keypoints[num1]['x'] * width, keypoints[num1]['y'] * height), (keypoints[num2]['x'] * width, keypoints[num2]['y'] * height), (keypoints[num3]['x'] * width, keypoints[num3]['y'] * height)
        angle = calculate_angle(a, b, c)
        draw_angle_arc(frame, b, a, c, angle, radius=25, color=col)
    
    def draw_line_between_points(num1, num2):
        p1 = (int(keypoints[num1]['x'] * width), int(keypoints[num1]['y'] * height))
        p2 = (int(keypoints[num2]['x'] * width), int(keypoints[num2]['y'] * height))
        cv2.line(frame, p1, p2, (180, 180, 180), 2)

    def angle_difference(num1, num2, num3, ref_angle):
        a, b, c = (keypoints[num1]['x'] * width, keypoints[num1]['y'] * height), (keypoints[num2]['x'] * width, keypoints[num2]['y'] * height), (keypoints[num3]['x'] * width, keypoints[num3]['y'] * height)
        actual_angle = calculate_angle(a, b, c)
        diff = (ref_angle - actual_angle + 180) % 360 - 180
        return diff
    
    def angle(num1, num2, num3):
        a, b, c = (keypoints[num1]['x'] * width, keypoints[num1]['y'] * height), (keypoints[num2]['x'] * width, keypoints[num2]['y'] * height), (keypoints[num3]['x'] * width, keypoints[num3]['y'] * height)
        actual_angle = calculate_angle(a, b, c)
        return actual_angle
    
    Y = (0, 255, 255)
    G = (0, 255, 0)
    R = (0, 0, 255)

    def get_colour_from_angle_diff(angle_diff, forgiveness: int = 30):
        if angle_diff <= forgiveness / 2:
            return G
        elif angle_diff <= forgiveness:
            return Y
        else:
            return R


    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = pose_data[frame_number].get("keypoints")

        frame_ratio = frame_number / frame_count
        sample_poses = load_reference_poses(classification, sample_path)
        best_reference_frame = find_best_reference_frame(sample_poses, frame_ratio)




        if len(keypoints) == 33:
            left_elbow_difference = angle_difference(12, 14, 16, best_reference_frame.get('angles').get('left_elbow_angle'))
            right_elbow_difference = angle_difference(11, 13, 15, best_reference_frame.get('angles').get('right_elbow_angle'))
            left_knee_difference = angle_difference(24, 26, 28, best_reference_frame.get('angles').get('left_knee_flexion'))
            right_knee_difference = angle_difference(23, 25, 27, best_reference_frame.get('angles').get('right_knee_flexion'))
            left_hip_difference = angle_difference(12, 24, 26, best_reference_frame.get('angles').get('left_hip_abduction'))
            right_hip_difference = angle_difference(11, 13, 25, best_reference_frame.get('angles').get('right_hip_abduction'))

            draw_point(0)

            for i in range(11, 33, 1):
                draw_point(i)

            draw_arc(12, 14, 16, get_colour_from_angle_diff(left_elbow_difference))
            draw_arc(11, 13, 15, get_colour_from_angle_diff(right_elbow_difference))
            draw_arc(24, 26, 28, get_colour_from_angle_diff(left_knee_difference))
            draw_arc(23, 25, 27, get_colour_from_angle_diff(right_knee_difference))
            draw_arc(12, 24, 26, get_colour_from_angle_diff(left_hip_difference))
            draw_arc(11, 23, 25, get_colour_from_angle_diff(right_hip_difference))

            draw_line_between_points(0, 11)
            draw_line_between_points(0, 12)
            draw_line_between_points(12, 14)
            draw_line_between_points(14, 16)
            draw_line_between_points(12, 24)
            draw_line_between_points(24, 26)
            draw_line_between_points(26, 28)
            draw_line_between_points(28, 30)
            draw_line_between_points(30, 32)
            draw_line_between_points(32, 28)
            draw_line_between_points(11, 12)
            draw_line_between_points(23, 24)
            draw_line_between_points(11, 23)
            draw_line_between_points(11, 13)
            draw_line_between_points(13, 15)
            draw_line_between_points(23, 25)
            draw_line_between_points(25, 27)
            draw_line_between_points(27, 29)
            draw_line_between_points(29, 31)
            draw_line_between_points(27, 31)
        else:
            print("skipping ellipse")
        

        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()
    print("Video processing complete and saved to", output_path)

    try:
        text = get_ai_feedback(classification, right_knee_difference, right_elbow_difference)
    except Exception as e:
        print(e)
        text = ""
    return text

import math



def calculate_angle(a, b, c):
    """Calculate angle (in degrees) at point b between a and c."""
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    cosine_angle = (ba[0]*bc[0] + ba[1]*bc[1]) / (math.hypot(*ba) * math.hypot(*bc) + 1e-6)
    angle = math.acos(min(1.0, max(-1.0, cosine_angle)))  # Clamp to avoid NaN
    return math.degrees(angle)

def draw_angle_arc(frame, b, a, c, angle, radius=30, color=(0, 255, 0), thickness=2):
    """Draw an arc representing the angle at point b between a and c."""

    # Convert points to numpy for vector math
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Vectors from b to a and b to c
    ba = a - b
    bc = c - b

    # Calculate starting angle using atan2
    start_angle = math.degrees(math.atan2(ba[1], ba[0]))
    end_angle = math.degrees(math.atan2(bc[1], bc[0]))

    # Fix direction: ensure we sweep in the correct direction
    arc_angle = (end_angle - start_angle) % 360
    if arc_angle > 180:
        arc_angle -= 360  # Sweep the shorter way

    # Draw the arc
    center = (int(b[0]), int(b[1]))
    cv2.ellipse(frame, center, (radius, radius), 0, start_angle, start_angle + arc_angle, color, thickness)

from pathlib import Path

def load_reference_poses(pose_type: int, sample_path: str):
    """
    Load the reference poses from the best-keypoints folder
    Returns a list of frames with their angles
    """
    pose_types = {0: "En Garde", 1: "Fleche", 2: "Lunge", 3: "Step"}
    
    all_frames = []
    with open(os.path.join(sample_path, pose_types[pose_type] + ".json"), 'r') as f:
        data = json.load(f)
        frames = data if isinstance(data, list) else [data]
        all_frames.extend(frames)
    
    all_frames.sort(key=lambda x: x.get('frame', 0) if isinstance(x, dict) else 0)
    return all_frames

def find_best_reference_frame(reference_frames, progress_ratio):
    """
    Find the best matching reference frame based on progress through the movement
    progress_ratio: float between 0 and 1 indicating progress through the movement
    """
    target_idx = int(progress_ratio * (len(reference_frames) - 1))
    return reference_frames[target_idx]

import openai

def get_ai_feedback(classification, kneeangle, elbowangle):
    api_key = os.getenv("LungeLearnApiKey")

    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Set the API key
    openai.api_key = api_key

    # Make a request to the ChatGPT model
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a fencing expert. Answer this question, rouhgly 50 words. First explain to me what you know based off the info I give you, then explain how I can improve. If there is not enough information, make up details. Try to back it up with evidence eg 'as evidenced by this angle'. Act as though you can see it directly."},
            {"role": "user", "content": f"I am practicing fencing doing the pose: {classification}. My knee angle is {kneeangle}. My elbow angle is {elbowangle}. Explain how I can improve, giving evidence in the form of angles."}
        ]
    )

    # Print the model's reply
    print(response["choices"][0]["message"]["content"])

    return response["choices"][0]["message"]["content"]