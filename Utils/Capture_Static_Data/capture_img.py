import cv2
import os
import mediapipe as mp
import numpy as np
import math
import time
from Constants.constants import id_cam, STATIC_DATA_WITH_OUT_P_DIR
# Inicializar Mediapipe Hands para detección y procesamiento de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  # Detectar hasta una mano
mp_drawing = mp.solutions.drawing_utils  # Utilidades para dibujar las marcas de Mediapipe

# Inicializar la cámara
id_camara = cv2.VideoCapture(id_cam)
offset = 20  # Desplazamiento para ajustar la caja delimitadora
img_out_size = 224  # Tamaño de la imagen de salida
dir = os.path.join(STATIC_DATA_WITH_OUT_P_DIR, "A")  # Cambiar la letra a guardar
img_count = 0  # Contador para las imágenes guardadas

while True:
    success, imagen_capturada = id_camara.read()  # Capturar una imagen de la cámara
    if not success:
        break

    # Convertir la imagen de BGR a RGB
    imagen_rgb = cv2.cvtColor(imagen_capturada, cv2.COLOR_BGR2RGB)
    resultados = hands.process(imagen_rgb)  # Procesar la imagen para detectar manos

    if resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            # Obtener la caja delimitadora de la mano
            x_min, y_min, x_max, y_max = float('inf'), float('inf'), float('-inf'), float('-inf')
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * imagen_capturada.shape[1]), int(lm.y * imagen_capturada.shape[0])
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y

            # Ajustar la caja delimitadora con un desplazamiento (offset)
            x_min = max(0, x_min - offset)
            y_min = max(0, y_min - offset)
            x_max = min(imagen_capturada.shape[1], x_max + offset)
            y_max = min(imagen_capturada.shape[0], y_max + offset)

            # Recortar y redimensionar la imagen de la mano
            img_mano_recortada = imagen_capturada[y_min:y_max, x_min:x_max]
            img_blanco = np.ones((img_out_size, img_out_size, 3), np.uint8) * 255  # Crear un lienzo blanco

            # Ajustar la imagen recortada al lienzo blanco
            h, w, _ = img_mano_recortada.shape
            ajuste_lienzo_blanco = h / w

            if ajuste_lienzo_blanco > 1:
                # Redimensionar si la altura es mayor que el ancho
                k = img_out_size / h
                nuevo_ancho = math.ceil(k * w)
                nueva_img_recortada = cv2.resize(img_mano_recortada, (nuevo_ancho, img_out_size))
                margen_ancho = math.ceil((img_out_size - nuevo_ancho) / 2)
                img_blanco[:, margen_ancho:nuevo_ancho + margen_ancho] = nueva_img_recortada
            else:
                # Redimensionar si el ancho es mayor que la altura
                k = img_out_size / w
                nueva_altura = math.ceil(k * h)
                nueva_img_recortada = cv2.resize(img_mano_recortada, (img_out_size, nueva_altura))
                margen_altura = math.ceil((img_out_size - nueva_altura) / 2)
                img_blanco[margen_altura:nueva_altura + margen_altura, :] = nueva_img_recortada

            # Mostrar las imágenes recortadas
            cv2.imshow("Imagen_Recortada_Inicial", img_mano_recortada)
            cv2.imshow("Imagen_Recortada_Final", img_blanco)

    # Mostrar la imagen capturada
    cv2.imshow("Image", imagen_capturada)
    key = cv2.waitKey(1)
    if key == ord("s"):
        # Guardar la imagen recortada cuando se presiona 's'
        img_count += 1
        cv2.imwrite(f'{dir}/Image_{time.time()}.jpg', img_blanco)
        print(img_count)
    elif key == ord("q"):
        # Salir del bucle cuando se presiona 'q'
        break

# Liberar la cámara y cerrar las ventanas
id_camara.release()
cv2.destroyAllWindows()
