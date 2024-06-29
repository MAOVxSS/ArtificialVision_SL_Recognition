import os
import cv2

# PATHS ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Raíz del proyecto
# Dynamic Model----------
DYNAMIC_DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, "Dynamic_Data"))
DYNAMIC_MODEL_DIR = os.path.abspath(os.path.join(ROOT_DIR, "Model", "Generated_Models"))
FRAME_ACTIONS_PATH = os.path.abspath(os.path.join(ROOT_DIR, "Frame_Actions"))
# ------------------------
# Static Model---------
STATIC_DATA_PROCESSED_DIR = os.path.abspath(os.path.join(ROOT_DIR, "Static_Data",
                                                         "Processed"))
STATIC_DATA_WITH_OUT_P_DIR = os.path.abspath(os.path.join(ROOT_DIR, "Static_Data",
                                                          "With_Out_Processed"))
STATIC_MODEL_DIR = os.path.abspath(os.path.join(ROOT_DIR, "Model", "Generated_Models"))
STATIC_MODEL_H5_DIR = os.path.abspath(os.path.join(ROOT_DIR, "Static_Data", "Data_H5"))
STATIC_MODEL_v3_DIR = os.path.abspath(os.path.join(ROOT_DIR, "Static_Data", "KeyPoints", "keypoints.csv"))
STATIC_MODEL_v3_LABELS_DIR = os.path.abspath(
    os.path.join(ROOT_DIR, "Static_Data", "KeyPoints", "keypoints_labels.csv"))
STATIC_MODEL_v3_TFLITE = os.path.abspath(
    os.path.join(ROOT_DIR, "Model", "Generated_Models", "keypoint_classifier.tflite"))

# ------------------------
# VARIABLES |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# Dynamic_Model-----------
MAX_LENGTH_FRAMES = 15  # Cantidad de frames para entrenamiento dinámico
LENGTH_KEYPOINTS = 63  # Longitud de los keypoints
MIN_LENGTH_FRAMES = 5
MODEL_NAME = f"actions_{MAX_LENGTH_FRAMES}.keras"
# ------------------------
# Static Model------------
labels = ["A", "B", "C", "D", "E", "F", "G",
          "H", "I", "L", "M", "N", "O", "P",
          "R", "S", "T", "U", "V", "W", "Y"]
# ------------------------
# VIDEO PARAMETERS
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SIZE = 1.5
FONT_POS = (5, 30)
id_cam = 1  # Se modifica cada vez que se usa una camara o otra
