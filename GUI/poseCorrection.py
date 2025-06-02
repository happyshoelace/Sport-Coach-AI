import json
from pathlib import Path
import pprint

def load_reference_poses(pose_type: int):
    """
    Load the reference poses from the best-keypoints folder
    Returns a list of frames with their angles
    """
    # pose_types = {0: "Fleche", 1: "Lunge", 2: "Step", 3: "En Garde"}
    pose_types = {0: "En Garde", 1: "Fleche", 2: "Lunge", 3: "Step"}
    
    all_frames = []
    with open(Path("samples", pose_types[pose_type]), 'r') as f:
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

# med pipe is already one frame
def poseCorrection(mediapipe_pose_data: json, pose_type: int, frame_number: int, total_frames: int, forgiveness: int = 10):
    # Load reference poses if not already loaded
    if not hasattr(poseCorrection, 'reference_poses'):
        poseCorrection.reference_poses = {}
    if pose_type not in poseCorrection.reference_poses:
        poseCorrection.reference_poses[pose_type] = load_reference_poses(pose_type)
    
    reference_frames = poseCorrection.reference_poses[pose_type]
    if not reference_frames:
        return None
    
    # Calculate progress through the movement (0 to 1)
    progress_ratio = frame_number / total_frames
    
    # Get the reference frame that best matches our current progress
    reference_frame = find_best_reference_frame(reference_frames, progress_ratio)
    
    # Extract angles from the current frame
    current_angles = mediapipe_pose_data['angles']
    reference_angles = reference_frame['angles']

    angle_keys_to_compare = [
        "left_elbow_angle",
        "right_elbow_angle",
        "left_knee_flexion",
        "right_knee_flexion",
        "torso_lean",
        "inter_arm_angle"
    ]

    differenceVals = [
        abs(current_angles[key] - reference_angles[key])
        for key in angle_keys_to_compare
    ]
    
    differenceRanking = []
    for i in differenceVals:
        if i <= forgiveness/2:
            differenceRanking.append("green")
        elif i <= forgiveness:
            differenceRanking.append("yellow")
        else:
            differenceRanking.append("red")
    
    return differenceVals, differenceRanking