import cv2
import mediapipe as mp
import numpy as np
import math
import os
import keras
from Constants.constants import id_cam, labels, STATIC_MODEL_DIR
# Inicializar Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Inicializar la cámara
cap = cv2.VideoCapture(id_cam)  # Ajustar si se cambia la camara
offset = 20  # Desplazamiento para ajustar la caja delimitadora
img_size = 224  # Tamaño de la imagen de salida

# Cargar el modelo
model_path = os.path.join(STATIC_MODEL_DIR, "best_model_v2.keras")
model = keras.models.load_model(model_path)

def preprocess_image(img_hand):
    img_white = np.ones((img_size, img_size, 3), np.uint8) * 255  # Crear un lienzo blanco

    h, w, _ = img_hand.shape
    aspect_ratio = h / w

    if aspect_ratio > 1:
        # Redimensionar si la altura es mayor que el ancho
        k = img_size / h
        new_width = math.ceil(k * w)
        resized_img = cv2.resize(img_hand, (new_width, img_size))
        width_margin = (img_size - new_width) // 2
        img_white[:, width_margin:width_margin + new_width] = resized_img
    else:
        # Redimensionar si el ancho es mayor que la altura
        k = img_size / w
        new_height = math.ceil(k * h)
        resized_img = cv2.resize(img_hand, (img_size, new_height))
        height_margin = (img_size - new_height) // 2
        img_white[height_margin:height_margin + new_height, :] = resized_img

    return img_white

while True:
    success, img = cap.read()  # Capturar una imagen de la cámara
    if not success:
        break

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir la imagen de BGR a RGB
    results = hands.process(rgb_img)  # Procesar la imagen para detectar manos

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Obtener la caja delimitadora de la mano
            x_min, y_min, x_max, y_max = float('inf'), float('inf'), float('-inf'), float('-inf')
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y

            # Ajustar la caja delimitadora con un offset
            x_min = max(0, x_min - offset)
            y_min = max(0, y_min - offset)
            x_max = min(img.shape[1], x_max + offset)
            y_max = min(img.shape[0], y_max + offset)

            # Recortar y redimensionar la imagen de la mano
            img_hand = img[y_min:y_max, x_min:x_max]
            img_white = preprocess_image(img_hand)

            # Normalizar la imagen
            img_white = img_white / 255.0

            # Expandir las dimensiones para que coincidan con el tamaño de entrada esperado del modelo
            img_white = np.expand_dims(img_white, axis=0)

            # Realizar la predicción
            prediction = model.predict(img_white)
            index = np.argmax(prediction)
            predicted_label = labels[index]

            # Mostrar la predicción en la imagen
            cv2.putText(img, f'Prediction: {predicted_label}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # Mostrar la imagen capturada
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
