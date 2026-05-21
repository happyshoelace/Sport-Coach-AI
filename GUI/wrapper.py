import os
from classifier import (save_json, 
                        remove_facial_keypoints_single_json, 
                        normalize_single_json, 
                        process_single_json_sequence, 
                        fill_and_prune_single_json, 
                        predict_windows_from_json,
                        process_single_json_with_angles_in_place, get_prediction_probability_and_index)
from draw_keypoints import draw_points

def video_name_to_predictions(video_name, hand):
    # save_json("./static/uploads", "IMG_0216_00000641_flipped.mov", "right", "./static/uploads")
    save_json("static/uploads", video_name.split('/')[-1], hand, "static/uploads")
    new_name = video_name.split('/')[-1] + ".json"

    remove_facial_keypoints_single_json(os.path.join("static/uploads", new_name))

    normalize_single_json(os.path.join("static/uploads", new_name))

    process_single_json_sequence(os.path.join("static/uploads", new_name))

    fill_and_prune_single_json(os.path.join("static/uploads", new_name))
   

    process_single_json_with_angles_in_place(os.path.join("static/uploads", new_name))
    
    total_frame_predictions = predict_windows_from_json(
        json_file_path=os.path.join("static/uploads", new_name),
        model_path="model.keras"
    )
   

    probability, index = get_prediction_probability_and_index(total_frame_predictions)
    print("Get Prediction Probability and Index Success")

    print("Done!", probability)

    return probability, index, total_frame_predictions

def save_video_with_keypoints(input_file, output_file):
    draw_points(os.path.join("static/uploads", input_file.split('/')[-1]), "static/uploads", output_file)
