import os
import pandas as pd
import numpy as np
import cv2
from mediapipe.python.solutions.hands import Hands
from Constants.constants import DATA_PATH, ROOT_PATH


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convierte la imagen a RGB
    image.flags.writeable = False  # Marca la imagen como no escribible
    results = model.process(image)  # Realiza la detección con MediaPipe
    image.flags.writeable = True  # Marca la imagen como escribible
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convierte la imagen de nuevo a BGR
    return image, results


def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.multi_hand_landmarks[0].landmark]).flatten() if results.multi_hand_landmarks else np.zeros(
        21 * 3)
    return lh


def get_keypoints_from_images(model, path):
    """
    OBTENER KEYPOINTS DE LAS IMÁGENES
    Retorna la secuencia de keypoints a partir de una serie de imágenes en una carpeta.

    Args:
    model (Hands): Modelo de MediaPipe para detección de manos.
    path (str): Ruta a la carpeta que contiene las imágenes.

    Returns:
    np.array: Secuencia de keypoints para las imágenes.
    """
    kp_seq = []
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        frame = cv2.imread(img_path)
        _, results = mediapipe_detection(frame, model)
        kp_frame = extract_keypoints(results)
        kp_seq.append(kp_frame)
    return np.array(kp_seq)


def insert_keypoints_sequence(df, kp_seq):
    """
    INSERTAR KEYPOINTS EN EL DATAFRAME
    Inserta la secuencia de keypoints en el DataFrame.

    Args:
    df (pd.DataFrame): DataFrame en el cual se insertarán los keypoints.
    kp_seq (np.array): Secuencia de keypoints.

    Returns:
    pd.DataFrame: DataFrame con los keypoints insertados.
    """
    for frame, keypoints in enumerate(kp_seq):
        data = {'frame': frame + 1, 'keypoints': [keypoints]}
        df_keypoints = pd.DataFrame(data)
        df = pd.concat([df, df_keypoints], ignore_index=True)
    return df


def create_keypoints_from_directory(data_path, save_path):
    """
    CREAR KEYPOINTS PARA TODAS LAS LETRAS
    Recorre la carpeta de imágenes y guarda sus keypoints en `save_path`.

    Args:
    data_path (str): Ruta a la carpeta que contiene las imágenes organizadas por clase.
    save_path (str): Ruta donde se guardarán las carpetas con archivos HDF5 de keypoints.
    """
    # Inicializa el modelo de MediaPipe Hands
    with Hands() as hands_model:
        for label in os.listdir(data_path):
            label_path = os.path.join(data_path, label)
            if os.path.isdir(label_path):
                print(f'Procesando la letra "{label}"...')
                keypoints_sequence = get_keypoints_from_images(hands_model, label_path)

                # Crear un DataFrame vacío
                data = pd.DataFrame([])

                # Insertar keypoints en el DataFrame
                data = insert_keypoints_sequence(data, keypoints_sequence)

                # Crear una carpeta para la letra si no existe
                label_save_path = os.path.join(save_path, label)
                if not os.path.exists(label_save_path):
                    os.makedirs(label_save_path)

                # Guardar los keypoints en un archivo HDF5 (.h5) separado para cada letra
                hdf_path = os.path.join(label_save_path, f"{label}.h5")
                data.to_hdf(hdf_path, key="data", mode="w")
                print(f'Keypoints para "{label}" creados y guardados en: {hdf_path}')


if __name__ == "__main__":
    data_path = os.path.abspath(os.path.join(ROOT_PATH, "..", "Static_Data", "With_Out_Processed"))
    save_path = os.path.abspath(os.path.join(ROOT_PATH, "..", "Static_Data", "Data_H5"))

    # Verifica si la ruta de los datos existe
    if not os.path.exists(data_path):
        print(f"Error: la ruta {data_path} no existe.")
        exit(1)

    # Crea la ruta para guardar los archivos HDF5 si no existe
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Crea los keypoints y guarda en carpetas con archivos HDF5 separados por letra
    print(f'Guardando keypoints en: {save_path}')
    create_keypoints_from_directory(data_path, save_path)
    print("Keypoints creados y guardados!")
