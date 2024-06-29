import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import mediapipe as mp
from Constants.constants import STATIC_MODEL_v3_LABELS_DIR, id_cam

# Cargar el archivo de etiquetas predefinido
labels_df = pd.read_csv(STATIC_MODEL_v3_LABELS_DIR, header=None, index_col=0)  # Lee el archivo CSV de etiquetas
labels_dict = labels_df[1].to_dict()  # Convierte las etiquetas a un diccionario

# Inicializar MediaPipe Hands.
mp_hands = mp.solutions.hands  # Inicializa la solución de detección de manos de MediaPipe
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5)  # Configura la detección de manos
mp_drawing = mp.solutions.drawing_utils  # Inicializa las utilidades de dibujo de MediaPipe

# Cargar el modelo TFLite
interpreter = tf.lite.Interpreter(
    model_path='../Model/Generated_Models/keypoint_classifier.tflite')  # Carga el modelo TFLite
interpreter.allocate_tensors()  # Asigna tensores para el modelo

# Obtener detalles de entrada y salida del modelo
input_details = interpreter.get_input_details()  # Obtiene detalles de los tensores de entrada del modelo
output_details = interpreter.get_output_details()  # Obtiene detalles de los tensores de salida del modelo


def predict_keypoints(keypoints):
    # Preprocesar los puntos clave para la entrada del modelo
    # Convierte los puntos clave a un arreglo NumPy y los redimensiona
    input_data = np.array(keypoints, dtype=np.float32).reshape(1,
                                                               21 * 3)
    interpreter.set_tensor(input_details[0]['index'], input_data)  # Establece los datos de entrada del modelo
    interpreter.invoke()  # Ejecuta la inferencia del modelo
    output_data = interpreter.get_tensor(output_details[0]['index'])  # Obtiene los resultados de la inferencia
    return np.argmax(output_data), np.max(output_data)  # Devuelve la etiqueta predicha y la confianza


def main():
    cap = cv2.VideoCapture(id_cam)  # Inicia la captura de video
    while cap.isOpened():  # Bucle mientras la cámara esté abierta
        ret, frame = cap.read()  # Lee un frame de la cámara
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convierte la imagen de BGR a RGB

        # Procesar la imagen con MediaPipe Hands
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:  # Si se detectan manos
            for hand_landmarks in results.multi_hand_landmarks:  # Itera sobre cada mano detectada
                # Dibujar las conexiones de los puntos clave en la imagen
                mp_drawing.draw_landmarks(frame, hand_landmarks,
                                          mp_hands.HAND_CONNECTIONS)  # Dibuja los puntos clave de la mano

                # Obtener los puntos clave de la mano
                keypoints = np.array(
                    [[landmark.x, landmark.y, landmark.z] for landmark in
                     hand_landmarks.landmark]).flatten()  # Extrae y aplana los puntos clave

                # Predecir la letra con el modelo TFLite
                predicted_label_id, confidence = predict_keypoints(
                    keypoints)  # Predice la letra basada en los puntos clave
                predicted_label = labels_dict[predicted_label_id]  # Obtiene la etiqueta predicha

                # Dibujar la letra predicha en la imagen
                cv2.putText(frame, f'{predicted_label} ({confidence:.2f})',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)  # Escribe la etiqueta en la imagen

        # Mostrar la imagen con los puntos clave y la predicción
        cv2.imshow('Hand Keypoints', frame)  # Muestra la imagen en una ventana

        if cv2.waitKey(10) & 0xFF == ord('q'):  # Si se presiona 'q', sale del bucle
            break

    cap.release()  # Libera la cámara
    cv2.destroyAllWindows()  # Cierra todas las ventanas de OpenCV


if __name__ == "__main__":
    main()  # Llama a la función principal
