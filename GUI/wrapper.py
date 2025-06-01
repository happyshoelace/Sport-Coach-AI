import os
from classifier import (save_json, 
                        remove_facial_keypoints_single_json, 
                        normalize_single_json, 
                        process_single_json_sequence, 
                        fill_and_prune_single_json, 
                        predict_windows_from_json,
                        process_single_json_with_angles_in_place, get_prediction_probability_and_index, process_video)

def video_name_to_predictions(video_name, hand):
    # save_json("./static/uploads", "IMG_0216_00000641_flipped.mov", "right", "./static/uploads")
    save_json("./static/uploads", video_name, hand, "./static/uploads")
    new_name = f"{video_name.split('/')[-1].split('.')[0]}.json"

    remove_facial_keypoints_single_json(os.path.join("./static/uploads", new_name))

    normalize_single_json(os.path.join("./static/uploads", new_name))

    process_single_json_sequence(os.path.join("./static/uploads", new_name))

    fill_and_prune_single_json(os.path.join("./static/uploads", new_name))

    process_single_json_with_angles_in_place(os.path.join("./static/uploads", new_name))

    total_frame_predictions = predict_windows_from_json(
        json_file_path=os.path.join("./static/uploads", new_name),
        model_path="model.keras"
    )

    probability, index = get_prediction_probability_and_index(total_frame_predictions)

    print("prob", probability, "ind", index)

    print("Done!", probability)

    process_video(os.path.join("./static/uploads", video_name), os.path.join("./static/uploads", video_name + "_angles"), new_name)

    print("Saved vid") 

    return probability, index, total_frame_predictions