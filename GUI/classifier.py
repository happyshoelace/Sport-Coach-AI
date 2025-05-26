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
# from google.colab.patches import cv2_imshow
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
# from google.colab.patches import cv2_imshow
import datetime

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

    # Filter to only class 'person' (usually class 0)
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
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False)

    # Video file processing
    if file_name.lower().endswith(('.mp4', '.mov')):
        video_path = os.path.join(input_base_path, file_name)
        cap = cv2.VideoCapture(video_path)
        
        # Prepare video writer for output video
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_video_filename = f"{file_name}_{timestamp}_output.mp4"
        output_video_path = os.path.join(output_base_path, output_video_filename)
        
        # Get the video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))
        
        frame_data = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame with Mediapipe Pose
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(frame_rgb)
            keypoints = []

            # If pose landmarks are detected, draw keypoints and edges
            if pose_results.pose_landmarks:
                # Draw the landmarks
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Save keypoints to json
                for lm in pose_results.pose_landmarks.landmark:
                    keypoints.append({
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z,
                        'visibility': lm.visibility
                    })

            # Write the frame with drawn keypoints to output video
            out_video.write(frame)

            # Append data to frame_data for JSON output
            frame_data.append({
                'frame': frame_idx,
                'keypoints': keypoints,
                'dominantHand': dominant_hand
            })
            frame_idx += 1

        # Release resources
        cap.release()
        out_video.release()

        # Save JSON data
        output_json_filename = f"{file_name}_{timestamp}.json"
        output_json_path = os.path.join(output_base_path, output_json_filename)
        with open(output_json_path, 'w') as f:
            json.dump(frame_data, f, indent=2)

        print(f"Processed video: {video_path} → {output_video_path}")
        print(f"Processed JSON: {video_path} → {output_json_path}")

        return output_json_path, output_video_path