import json
def poseCorrection(mediapipe_pose_data: json, pose_type: int):
    """
    This function takes in a JSON object containing the mediapipe pose data and a pose type.
    It then returns a list of the difference values and a list of the difference rankings.

    It should be ran once per frame.
    """
    forgiveness = 10 # degrees of forgiveness

    # 0: fleche
    # 1: lunge
    # 2: step
    # 3: en garde
    match pose_type:
        case 0:
            domElbowAngleP = 0
            subElbowAngleP = 0
            rightKneeFlexionP = 0
            leftKneeFlexionP = 0
            torsoLeanP = 0
            interArmAngleP = 0
            
        case 1:
            domElbowAngleP = 0
            subElbowAngleP = 0
            rightKneeFlexionP = 0
            leftKneeFlexionP = 0
            torsoLeanP = 0
            interArmAngleP = 0

        case 2:
            domElbowAngleP = 0
            subElbowAngleP = 0
            rightKneeFlexionP = 0
            leftKneeFlexionP = 0
            torsoLeanP = 0
            interArmAngleP = 0

        case 3:
            domElbowAngleP = 0
            subElbowAngleP = 0
            rightKneeFlexionP = 0
            leftKneeFlexionP = 0
            torsoLeanP = 0
            interArmAngleP = 0
        case _:
            return None

    
    # Extract angles from the JSON data
    angles = mediapipe_pose_data['angles']
    
    # Get specific angles
    domElbowAngle = angles['right_elbow_angle'] if mediapipe_pose_data['dominantHand'] == 'Right' else angles['left_elbow_angle']
    subElbowAngle = angles['left_elbow_angle'] if mediapipe_pose_data['dominantHand'] == 'Right' else angles['right_elbow_angle']
    rightKneeFlexion = angles['right_knee_flexion']
    leftKneeFlexion = angles['left_knee_flexion']
    torsoLean = angles['torso_lean']
    interArmAngle = angles['inter_arm_angle']

    differenceVals = [abs(domElbowAngle - domElbowAngleP), abs(subElbowAngle - subElbowAngleP), 
    abs(rightKneeFlexion - rightKneeFlexionP), abs(leftKneeFlexion - leftKneeFlexionP), 
    abs(torsoLean - torsoLeanP), abs(interArmAngle - interArmAngleP)]
    
    differenceRanking = []
    
    for i in differenceVals:
        if i == 0:
            differenceRanking.append("green")
        elif i <= forgiveness:
            differenceRanking.append("yellow")
        else:
            differenceRanking.append("red")

    return differenceVals, differenceRanking