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
import mediapipe as mp
import pandas as pd
import matplotlib.pyplot as plt

def draw_points(input_file_path, output_folder, output_file_name):
    print("Drawing points on", input_file_path, "to", output_folder, output_file_name)
    video_path = input_file_path

    video_name = output_file_name
    output_folder = f"{video_name}_vid"

    os.makedirs(os.path.join(input_file_path, output_folder), exist_ok=True)
    print(os.path.join(input_file_path, output_folder))

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video for drawing")
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

    output_video_path = f"{video_name}_output.mp4"

    frame_files = sorted([
        os.path.join(output_folder, f)
        for f in os.listdir(output_folder)
        if f.endswith('.jpg')
    ])

    out = cv2.VideoWriter(
        output_video_path,
        # cv2.VideoWriter_fourcc(*'mp4v'),
        cv2.VideoWriter_fourcc(*'XVID'),
        24,
        (frame_width, frame_height)
    )

    for file_path in frame_files:
        frame = cv2.imread(file_path)
        out.write(frame)

    out.release()

    print("Saved video to", output_folder, output_file_name)