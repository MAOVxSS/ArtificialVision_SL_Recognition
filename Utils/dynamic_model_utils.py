import os
import cv2
from mediapipe.python.solutions.hands import HAND_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec
import numpy as np
from typing import NamedTuple
import pandas as pd


# Función para procesar la imagen con MediaPipe y detectar manos
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convierte la imagen de BGR a RGB
    image.flags.writeable = False  # Optimiza la velocidad de procesamiento
    results = model.process(image)  # Procesa la imagen con el modelo de MediaPipe
    image.flags.writeable = True  # Vuelve a permitir escritura en la imagen
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convierte la imagen de vuelta a BGR
    return image, results  # Devuelve la imagen procesada y los resultados


# Función para obtener acciones (letras) desde el directorio
def get_actions(path):
    out = []
    for action in os.listdir(path):
        name, ext = os.path.splitext(action)  # Divide el nombre y la extensión del archivo
        if ext == ".h5":  # Si la extensión es .h5, añade el nombre a la lista
            out.append(name)
    return out  # Devuelve la lista de nombres de acciones


# Función para crear una carpeta si no existe
def create_folder(path):
    if not os.path.exists(path):  # Si el directorio no existe
        os.makedirs(path)  # Crea el directorio


# Función para verificar si hay una mano detectada en los resultados
def there_hand(results: NamedTuple) -> bool:
    return results.multi_hand_landmarks is not None  # Devuelve True si se detectaron manos


# Función para dibujar puntos clave en la imagen
def draw_keypoints(image, results):
    if results.multi_hand_landmarks:  # Si se detectaron manos
        for hand_landmarks in results.multi_hand_landmarks:  # Itera sobre cada mano detectada
            draw_landmarks(
                image,
                hand_landmarks,
                HAND_CONNECTIONS,
                DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                # Especificaciones de dibujo para los puntos
                DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
                # Especificaciones de dibujo para las conexiones
            )


# Función para guardar frames en un directorio
def save_frames(frames, output_folder):
    for num_frame, frame in enumerate(frames):  # Itera sobre los frames
        frame_path = os.path.join(output_folder, f"{num_frame + 1}.jpg")  # Define la ruta del archivo
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA))  # Guarda el frame como imagen .jpg


# Función para extraer puntos clave de los resultados de MediaPipe
def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.multi_hand_landmarks[0].landmark]).flatten() if results.multi_hand_landmarks else np.zeros(
        21 * 3)  # Extrae y aplana los puntos clave de la primera mano detectada
    return lh  # Devuelve los puntos clave o un array de ceros si no hay manos


# Función para obtener secuencias de puntos clave desde un directorio
def get_keypoints(model, path):
    kp_seq = np.array([])  # Inicializa un array vacío para la secuencia de puntos clave
    for img_name in os.listdir(path):  # Itera sobre los nombres de los archivos en el directorio
        img_path = os.path.join(path, img_name)  # Define la ruta completa del archivo
        frame = cv2.imread(img_path)  # Lee la imagen
        _, results = mediapipe_detection(frame, model)  # Detecta manos en la imagen
        kp_frame = extract_keypoints(results)  # Extrae los puntos clave
        kp_seq = np.concatenate(
            [kp_seq, [kp_frame]] if kp_seq.size > 0 else [[kp_frame]])  # Añade los puntos clave a la secuencia
    return kp_seq  # Devuelve la secuencia de puntos clave


# Función para insertar una secuencia de puntos clave en un DataFrame
def insert_keypoints_sequence(df, n_sample: int, kp_seq):
    for frame, keypoints in enumerate(kp_seq):  # Itera sobre la secuencia de puntos clave
        data = {'sample': n_sample, 'frame': frame + 1, 'keypoints': [keypoints]}  # Crea un diccionario con los datos
        df_keypoints = pd.DataFrame(data)  # Convierte el diccionario en un DataFrame
        df = pd.concat([df, df_keypoints])  # Añade el DataFrame al original
    return df  # Devuelve el DataFrame actualizado


# Función para obtener secuencias y etiquetas desde los archivos de datos
def get_sequences_and_labels(actions, data_path):
    sequences, labels = [], []  # Inicializa listas para secuencias y etiquetas

    for label, action in enumerate(actions):  # Itera sobre las acciones con sus etiquetas
        hdf_path = os.path.join(data_path, f"{action}.h5")  # Define la ruta del archivo .h5
        data = pd.read_hdf(hdf_path, key='data')  # Lee los datos del archivo .h5

        for _, data_filtered in data.groupby('sample'):  # Agrupa los datos por muestras
            sequences.append(
                [fila['keypoints'] for _, fila in data_filtered.iterrows()])  # Añade la secuencia de puntos clave
            labels.append(label)  # Añade la etiqueta correspondiente

    return sequences, labels  # Devuelve las secuencias y etiquetas


# Función para guardar contenido en un archivo de texto
def save_txt(file_name, content):
    with open(file_name, 'w') as archivo:  # Abre el archivo en modo escritura
        archivo.write(content)  # Escribe el contenido en el archivo


# Función para formatear una letra y contar repeticiones
def format_letter(sent, letter, repe_sent):
    if len(letter) > 1:  # Si hay más de una letra en la lista
        if sent in letter[1]:  # Si la letra actual está en la siguiente letra
            repe_sent += 1  # Incrementa el contador de repeticiones
            letter.pop(0)  # Elimina la primera letra de la lista
            letter[0] = f"{sent} (x{repe_sent})"  # Formatea la letra con el contador de repeticiones
        else:
            repe_sent = 1  # Reinicia el contador de repeticiones
    return letter, repe_sent  # Devuelve la lista de letras y el contador de repeticiones
