import os

# Paths ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Raíz del proyecto
# Dynamic Model----------
dynamic_data_dir = os.path.abspath(os.path.join(root_dir, "Dynamic_Data"))
dynamic_model_dir = os.path.abspath(os.path.join(root_dir, "Model", "Generated_Models"))
frame_actions_path = os.path.abspath(os.path.join(root_dir, "Dynamic_Data", "Frame_Actions"))
dynamic_model_name = "dynamic_model.keras"
# ------------------------
# Static Model---------
static_data_processed_dir = os.path.abspath(os.path.join(root_dir, "Static_Data",
                                                         "Processed"))
static_data_with_out_p_dir = os.path.abspath(os.path.join(root_dir, "Static_Data",
                                                          "With_Out_Processed"))
static_model_dir = os.path.abspath(os.path.join(root_dir, "Model", "Generated_Models"))
static_model_h5_dir = os.path.abspath(os.path.join(root_dir, "Static_Data", "Data_H5"))
static_model_v3_keypoints_dir = os.path.abspath(os.path.join(root_dir, "Static_Data", "KeyPoints", "keypoints.csv"))
static_model_v3_labels_dir = os.path.abspath(
    os.path.join(root_dir, "Static_Data", "KeyPoints", "keypoints_labels.csv"))
static_model_v3_tflite = os.path.abspath(
    os.path.join(root_dir, "Model", "Generated_Models", "keypoint_classifier.tflite"))
static_model_name = "static_keypoint.keras"
# ------------------------
# Variables |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# Static Model------------
labels = ["A", "B", "C", "D", "E", "F", "G",
          "H", "I", "L", "M", "N", "O", "P",
          "R", "S", "T", "U", "V", "W", "Y"]
# ------------------------
# Video parameters
id_cam = 1  # Se modifica cada vez que se usa una camara o otra
