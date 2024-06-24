import os
import cv2
from mediapipe.python.solutions.hands import HAND_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec
import numpy as np
from typing import NamedTuple
import pandas as pd
import matplotlib.pyplot as plt

def mediapipe_detection(image, model):
    """
    DETECCIÓN CON MEDIAPIPE
    Convierte la imagen a RGB, la procesa con el modelo de mediapipe y devuelve los resultados.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False  # Optimiza la velocidad de procesamiento
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def get_actions(path):
    """
    OBTENER ACCIONES
    Retorna una lista de nombres de acciones que tienen archivos .h5 en el directorio dado.
    """
    out = []
    for action in os.listdir(path):
        name, ext = os.path.splitext(action)
        if ext == ".h5":
            out.append(name)
    return out


def create_folder(path):
    """
    CREAR CARPETA SI NO EXISTE
    Crea una carpeta en la ruta especificada si no existe.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def there_hand(results: NamedTuple) -> bool:
    """
    DETECTAR MANO
    Verifica si hay detecciones de manos en los resultados.
    """
    return results.multi_hand_landmarks is not None


def draw_keypoints(image, results):
    """
    DIBUJAR KEYPOINTS
    Dibuja los puntos clave de la mano en la imagen.
    """
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            draw_landmarks(
                image,
                hand_landmarks,
                HAND_CONNECTIONS,
                DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
            )


def save_frames(frames, output_folder):
    """
    GUARDAR FOTOGRAMAS
    Guarda cada fotograma de una lista de fotogramas en la carpeta de salida especificada.
    """
    for num_frame, frame in enumerate(frames):
        frame_path = os.path.join(output_folder, f"{num_frame + 1}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA))


def extract_keypoints(results):
    """
    EXTRAER KEYPOINTS
    Extrae los puntos clave de la primera mano detectada en los resultados.
    """
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.multi_hand_landmarks[0].landmark]).flatten() if results.multi_hand_landmarks else np.zeros(
        21 * 3)
    return lh


def get_keypoints(model, path):
    """
    OBTENER KEYPOINTS DE LA MUESTRA
    Retorna la secuencia de puntos clave (keypoints) de las imágenes en el directorio dado.
    """
    kp_seq = np.array([])
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        frame = cv2.imread(img_path)
        _, results = mediapipe_detection(frame, model)
        kp_frame = extract_keypoints(results)
        kp_seq = np.concatenate([kp_seq, [kp_frame]] if kp_seq.size > 0 else [[kp_frame]])
    return kp_seq


def insert_keypoints_sequence(df, n_sample: int, kp_seq):
    """
    INSERTAR SECUENCIA DE KEYPOINTS AL DATAFRAME
    Agrega la secuencia de puntos clave (keypoints) de la muestra al DataFrame.
    """
    for frame, keypoints in enumerate(kp_seq):
        data = {'sample': n_sample, 'frame': frame + 1, 'keypoints': [keypoints]}
        df_keypoints = pd.DataFrame(data)
        df = pd.concat([df, df_keypoints])
    return df


def get_sequences_and_labels(actions, data_path):
    """
    OBTENER SECUENCIAS Y ETIQUETAS
    Retorna las secuencias de puntos clave (keypoints) y sus etiquetas correspondientes.
    """
    sequences, labels = [], []

    for label, action in enumerate(actions):
        hdf_path = os.path.join(data_path, f"{action}.h5")
        data = pd.read_hdf(hdf_path, key='data')

        for _, data_filtered in data.groupby('sample'):
            sequences.append([fila['keypoints'] for _, fila in data_filtered.iterrows()])
            labels.append(label)

    return sequences, labels


def save_txt(file_name, content):
    """
    GUARDAR TEXTO EN ARCHIVO
    Guarda el contenido en un archivo de texto con el nombre especificado.
    """
    with open(file_name, 'w') as archivo:
        archivo.write(content)


def format_letter(sent, letter, repe_sent):
    """
    FORMATEAR LETRAS
    Formatea las letras para agregar un contador de repeticiones si la letra ya existe en la lista.
    """
    if len(letter) > 1:
        if sent in letter[1]:
            repe_sent += 1
            letter.pop(0)
            letter[0] = f"{sent} (x{repe_sent})"
        else:
            repe_sent = 1
    return letter, repe_sent


# Función para binarizar la imagen entrante
def binarize_img(original_img, hsv_image):
    img_bin = np.zeros((original_img.shape[0], original_img.shape[1]))
    for i in range(hsv_image.shape[0]):
        for j in range(hsv_image.shape[1]):
            if (hsv_image[i, j, 0] > 107 and hsv_image[i, j, 0] < 127) and \
                    (hsv_image[i, j, 1] > 48 and hsv_image[i, j, 1] < 157) and \
                    (hsv_image[i, j, 2] > 56 and hsv_image[i, j, 2] < 186):
                img_bin[i, j] = 255  # Cambiar el valor a 255 para visualizar correctamente la imagen binarizada
    return img_bin

# Función para mostrar información extra sobre el entrenamiento de los modelos

def plot_history(history):
    # Graficar la precisión
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Precisión a través de las épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()

    # Graficar la pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida a través de las épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    plt.show()