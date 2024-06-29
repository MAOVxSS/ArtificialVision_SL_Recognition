import os
import numpy as np
import cv2
import mediapipe as mp
import pandas as pd
from Constants.constants import STATIC_MODEL_v3_LABELS_DIR, STATIC_MODEL_v3_DIR, id_cam

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                       min_detection_confidence=0.5)  # Configura la detección de manos
mp_drawing = mp.solutions.drawing_utils  # Inicializa las utilidades de dibujo de MediaPipe

# Cargar el archivo de etiquetas predefinido
labels_df = pd.read_csv(STATIC_MODEL_v3_LABELS_DIR, header=None, index_col=0)  # Lee el archivo CSV de etiquetas
labels_dict = labels_df[1].to_dict()  # Convierte las etiquetas a un diccionario

# Identificador de la letra a capturar
label_id = 20  # Cambia este valor según la letra que quieras capturar (0-20)


# Función para capturar keypoints utilizando MediaPipe
def capture_keypoints(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convierte la imagen de BGR a RGB
    results = model.process(image_rgb)  # Procesa la imagen para detectar manos
    if results.multi_hand_landmarks:  # Si se detectan manos
        hand_landmarks = results.multi_hand_landmarks[0]  # Toma la primera mano detectada
        keypoints = np.array([[landmark.x, landmark.y, landmark.z] for landmark in
                              hand_landmarks.landmark]).flatten()  # Extrae y aplana los puntos clave
        return keypoints, results.multi_hand_landmarks  # Devuelve los puntos clave y las marcas de la mano
    else:
        keypoints = np.zeros(21 * 3)  # 21 puntos con x, y, z (si no se detecta mano)
        return keypoints, None


# Función para guardar los datos
def save_data(keypoints, label, keypoints_file):
    data = np.hstack([label, keypoints])  # Concatenar la etiqueta con los puntos clave
    with open(keypoints_file, 'a') as f:
        np.savetxt(f, [data], delimiter=',')  # Guardar los datos en el archivo
    print(f"Data saved for label: {label}")  # Confirmación de guardado


# Función para contar muestras
def count_samples(keypoints_file):
    if os.path.exists(keypoints_file):  # Verifica si el archivo existe
        data = np.loadtxt(keypoints_file, delimiter=',')  # Carga los datos del archivo
        if data.ndim == 1:  # Solo hay una fila en el archivo
            labels = [int(data[0])]
        else:
            labels = data[:, 0].astype(int)  # Extrae las etiquetas de las filas
        label_counts = {label_id: list(labels).count(label_id) for label_id in
                        set(labels)}  # Cuenta las muestras por etiqueta
        for label_id, count in label_counts.items():
            print(f"Label {label_id} ({labels_dict[label_id]}): {count} samples")  # Muestra el conteo de muestras
    else:
        print("No samples captured yet.")  # Mensaje si no hay muestras capturadas


# Función principal
def main():
    print("Press 's' to start capturing data and 'q' to quit and save data.")  # Instrucciones para el usuario
    label = labels_dict[label_id]  # Obtener la etiqueta correspondiente al ID
    print(f"Capturing data for label: {label} (ID: {label_id})")  # Mensaje de inicio de captura

    cap = cv2.VideoCapture(id_cam)  # Inicia la captura de video desde la cámara
    capturing = False  # Bandera para controlar la captura
    while cap.isOpened():
        ret, frame = cap.read()  # Lee un frame de la cámara
        if not ret:
            break

        keypoints, hand_landmarks = capture_keypoints(frame, hands)  # Captura los puntos clave de la mano
        if hand_landmarks:  # Si se detectaron manos
            for hand_landmark in hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmark,
                                          mp_hands.HAND_CONNECTIONS)  # Dibuja las marcas de la mano

        # Mostrar la imagen con los puntos clave
        cv2.imshow('Hand Keypoints', frame)  # Muestra la imagen en una ventana

        key = cv2.waitKey(10)  # Espera por una tecla
        if key & 0xFF == ord('s'):
            capturing = True  # Inicia la captura de datos
            print(f"Started capturing data for label: {label} (ID: {label_id})")

        if key & 0xFF == ord('q'):
            break  # Sale del bucle si se presiona 'q'

        if capturing:
            save_data(keypoints, label_id, STATIC_MODEL_v3_DIR)  # Guarda los datos capturados
            print(f"Captured data for label: {label} (ID: {label_id})")
            capturing = False  # Resetea la bandera de captura para evitar múltiples capturas sin presionar 's'
            count_samples(STATIC_MODEL_v3_DIR)  # Muestra el conteo de muestras

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()  # Llama a la función principal
