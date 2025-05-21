import mediapipe as mp

import cv2

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import cv2
from google.colab.patches import cv2_imshow
import mediapipe as mp
import pandas as pd
import matplotlib.pyplot as plt

# Get list of all files and directories

def create_windows(frames, window_size=50, stride=1):
    windows = []
    for start in range(0, len(frames) - window_size + 1, stride):
        window = frames[start:start + window_size]
        windows.append(window)
    return np.array(windows)

def preprocess_sample(frames_data, window_size):
    frames = load_json_file(frame_data)
    print(frames, len(frames))
    if len(frames) < window_size:
        frames = pad_frames(frames, window_size)
        windows = [frames]
    else:
        windows = create_windows(frames, window_size, 1)

    return np.array(windows)

def load_json_file(frame_data_total):
    frames = []
    for frame_data in frame_data_total:
        # print("frame:", frame_data)
        keypoints = frame_data["keypoints"]
        frame = []
        for kp in keypoints:
            frame.extend([kp["x"], kp["y"], kp["z"], kp["visibility"]])
        if len(frame) == 132:
            frames.append(frame)
    return frames

def pad_frames(frames, window_size, frame_size=132):

    frames = np.array(frames)

    # Reshape to (n_frames, frame_size)
    frames = frames.reshape(-1, frame_size)
    num_frames = frames.shape[0]

    if num_frames >= window_size:
        return frames[:window_size]

    # Padding
    pad_length = window_size - num_frames
    padding = np.zeros((pad_length, frame_size), dtype=frames.dtype)
    return np.vstack((frames, padding))

class_folders = ['Fleche', 'Lunge', 'Step', 'En Garde']

le = LabelEncoder()
le.fit(class_folders)

model = tf.keras.models.load_model("model_pose_fence.h5")

# Example of bad video (mediapipe detects keypoints but wrong pose)
# video_path = './test_lunge_cropped.mp4'
# Example of good video (mediapipe detects keypoints, right pose)
# video_path = './Training Data New/Step/Right/R-Forward/122.mov'

def classify(vid_name):
    video_name = os.path.splitext(os.path.basename(vid_name))[0]
    output_folder = f"{video_name}_vid"

    os.makedirs(output_folder, exist_ok=True)

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False)

    cap = cv2.VideoCapture(vid_name)

    if not cap.isOpened():
        print("Error: Could not open video.")
        input()

    count = 0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_data = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
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
            frame_data.append({
                'frame': frame_idx,
                'keypoints': keypoints
            })
            frame_idx += 1

            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
                )

            frame_output_path = os.path.join(output_folder, f"{video_name}_{frame_idx:04d}.jpg")
            cv2.imwrite(frame_output_path, frame)

        except Exception as e:
            print(f"Error on frame {count}: {e}")
            break
        count += 1


    cap.release()

    # Output video path
    output_video_path = f"{video_name}_output.mp4"

    # Get list of saved frame paths and sort
    frame_files = sorted([
        os.path.join(output_folder, f)
        for f in os.listdir(output_folder)
        if f.endswith('.jpg')
    ])

    # TODO: change from 30 to something dynamic
    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,
        (frame_width, frame_height)
    )

    for file_path in frame_files:
        frame = cv2.imread(file_path)
        out.write(frame)

    out.release()
    print(f"Output video saved to {output_video_path}")


    # Load and preprocess sample
    sample_windows = preprocess_sample(frame_data, 50)
    # print(sample_windows)

    # Predict
    predictions = model.predict(sample_windows)  # shape: (num_windows, num_classes)

    # Average predictions across windows
    avg_prediction = np.mean(predictions, axis=0)

    # Decode class
    predicted_class_index = np.argmax(avg_prediction)
    predicted_class_label = le.inverse_transform([predicted_class_index])[0]
    
    return (predicted_class_label, avg_prediction[predicted_class_index])
