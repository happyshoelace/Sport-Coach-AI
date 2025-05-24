import json
from pathlib import Path
import pprint

def load_reference_poses(pose_type: int):
    """
    Load the reference poses from the best-keypoints folder
    Returns a list of frames with their angles
    """
    pose_types = {0: "Fleche", 1: "Lunge", 2: "Step", 3: "En Garde"}
    pose_folder = Path("best-keypoints") / pose_types[pose_type]
    
    all_frames = []
    for json_file in pose_folder.glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            # If the data is a single frame, wrap it in a list
            frames = data if isinstance(data, list) else [data]
            all_frames.extend(frames)
    
    # Sort frames by frame number if available
    all_frames.sort(key=lambda x: x.get('frame', 0) if isinstance(x, dict) else 0)
    return all_frames

def find_best_reference_frame(current_frame, reference_frames, progress_ratio):
    """
    Find the best matching reference frame based on progress through the movement
    progress_ratio: float between 0 and 1 indicating progress through the movement
    """
    target_idx = int(progress_ratio * (len(reference_frames) - 1))
    return reference_frames[target_idx]

def poseCorrection(mediapipe_pose_data: json, pose_type: int, frame_number: int, total_frames: int, forgiveness: int = 10):
    """
    This function takes in a JSON object containing the mediapipe pose data and a pose type.
    It compares the current frame against the reference poses and returns difference values and rankings.
    
    Args:
        mediapipe_pose_data: JSON object with pose data for current frame
        pose_type: int indicating pose type (0: fleche, 1: lunge, 2: step, 3: en garde)
        frame_number: current frame number in the sequence
        total_frames: total number of frames in the input sequence
        forgiveness: int indicating the number of degrees of forgiveness
    
    Returns:
        tuple: (difference_values, difference_rankings)
    """


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
    reference_frame = find_best_reference_frame(mediapipe_pose_data, reference_frames, progress_ratio)
    
    # Extract angles from the current frame
    angles = mediapipe_pose_data['angles']
    dominant_hand = mediapipe_pose_data['dominantHand']

    if angles == {}:
        return None, None
    

    # Get current frame angles
    domElbowAngle = angles['right_elbow_angle'] if dominant_hand == 'Right' else angles['left_elbow_angle']
    subElbowAngle = angles['left_elbow_angle'] if dominant_hand == 'Right' else angles['right_elbow_angle']
    rightKneeFlexion = angles['right_knee_flexion']
    leftKneeFlexion = angles['left_knee_flexion']
    torsoLean = angles['torso_lean']
    interArmAngle = angles['inter_arm_angle']
    
    # Get reference frame angles
    ref_angles = reference_frame['angles']
    ref_dominant_hand = reference_frame['dominantHand']
    
    
    # Get reference angles
    ref_domElbowAngle = ref_angles['right_elbow_angle'] if ref_dominant_hand == 'Right' else ref_angles['left_elbow_angle']
    ref_subElbowAngle = ref_angles['left_elbow_angle'] if ref_dominant_hand == 'Right' else ref_angles['right_elbow_angle']
    ref_rightKneeFlexion = ref_angles['right_knee_flexion']
    ref_leftKneeFlexion = ref_angles['left_knee_flexion']
    ref_torsoLean = ref_angles['torso_lean']
    ref_interArmAngle = ref_angles['inter_arm_angle']
    
    # Calculate differences
    differenceVals = [
        abs(domElbowAngle - ref_domElbowAngle),
        abs(subElbowAngle - ref_subElbowAngle),
        abs(rightKneeFlexion - ref_rightKneeFlexion),
        abs(leftKneeFlexion - ref_leftKneeFlexion),
        abs(torsoLean - ref_torsoLean),
        abs(interArmAngle - ref_interArmAngle)
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
