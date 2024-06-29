import os
import pandas as pd
import numpy as np
import cv2
from mediapipe.python.solutions.hands import Hands
from Utils.dynamic_model_utils import get_keypoints, insert_keypoints_sequence, mediapipe_detection
from Constants.constants import FRAME_ACTIONS_PATH, DYNAMIC_DATA_DIR


def get_keypoints(model, path):
    """
    OBTENER KEYPOINTS DE LA MUESTRA
    Retorna la secuencia de keypoints de la muestra a partir de una serie de imágenes.

    Args:
    model (Hands): Modelo de MediaPipe para detección de manos.
    path (str): Ruta a la carpeta que contiene las imágenes.

    Returns:
    np.array: Secuencia de keypoints para las imágenes de la muestra.
    """
    kp_seq = np.array([])
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        frame = cv2.imread(img_path)
        _, results = mediapipe_detection(frame, model)
        kp_frame = extract_keypoints(results)
        kp_seq = np.concatenate([kp_seq, [kp_frame]] if kp_seq.size > 0 else [[kp_frame]])
    return kp_seq


def extract_keypoints(results):
    """
    EXTRAER KEYPOINTS DE LOS RESULTADOS DE MEDIAPIPE
    Extrae los keypoints de los landmarks de la mano si existen, de lo contrario retorna un arreglo de ceros.

    Args:
    results: Resultados de la detección de MediaPipe.

    Returns:
    np.array: Keypoints extraídos o un arreglo de ceros si no hay detección.
    """
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.multi_hand_landmarks[0].landmark]).flatten() if results.multi_hand_landmarks else np.zeros(
        21 * 3)
    return lh


def insert_keypoints_sequence(df, n_sample: int, kp_seq):
    """
    INSERTAR KEYPOINTS EN EL DATAFRAME
    Inserta la secuencia de keypoints en el DataFrame.

    Args:
    df (pd.DataFrame): DataFrame en el cual se insertarán los keypoints.
    n_sample (int): Número de la muestra.
    kp_seq (np.array): Secuencia de keypoints.

    Returns:
    pd.DataFrame: DataFrame con los keypoints insertados.
    """
    for frame, keypoints in enumerate(kp_seq):
        data = {'sample': n_sample, 'frame': frame + 1, 'keypoints': [keypoints]}
        df_keypoints = pd.DataFrame(data)
        df = pd.concat([df, df_keypoints])
    return df


def create_keypoints(frames_path, save_path):
    """
    CREAR KEYPOINTS PARA UNA LETRA
    Recorre la carpeta de frames de la palabra y guarda sus keypoints en `save_path`.

    Args:
    frames_path (str): Ruta a la carpeta que contiene los frames de la palabra.
    save_path (str): Ruta donde se guardará el archivo HDF5 con los keypoints.
    """
    data = pd.DataFrame([])

    # Inicializa el modelo de MediaPipe Hands
    with Hands() as hands_model:
        for n_sample, sample_name in enumerate(os.listdir(frames_path), 1):
            sample_path = os.path.join(frames_path, sample_name)
            keypoints_sequence = get_keypoints(hands_model, sample_path)
            data = insert_keypoints_sequence(data, n_sample, keypoints_sequence)

    # Guarda los keypoints en un archivo HDF5 (.h5)
    data.to_hdf(save_path, key="data", mode="w")


if __name__ == "__main__":

    # Generar los keypoints de todas las palabras
    for word_name in os.listdir(FRAME_ACTIONS_PATH):
        word_path = os.path.join(FRAME_ACTIONS_PATH, word_name)
        hdf_path = os.path.join(DYNAMIC_DATA_DIR, f"{word_name}.h5")
        print(f'Creando keypoints de "{word_name}"...')
        create_keypoints(word_path, hdf_path)
        print(f"Keypoints creados!")
