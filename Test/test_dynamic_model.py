from mediapipe.python.solutions.hands import Hands
from tensorflow.keras.models import load_model
from Utils.utils import *
from Constants.constants import *


def evaluate_model(model, threshold=0.7):
    # Inicialización de variables
    count_frame = 0
    repe_sent = 1
    keypoint_sequence, sentence = [], []
    actions = get_actions(DYNAMIC_DATA_DIR)  # Obtiene las acciones posibles desde la ruta de datos

    with Hands() as hands_model:
        video = cv2.VideoCapture(id_cam)  # Inicia la captura de video desde la cámara

        while video.isOpened():
            _, frame = video.read()  # Lee un fotograma de la cámara

            # Realiza la detección con MediaPipe y extrae los puntos clave
            image, results = mediapipe_detection(frame, hands_model)
            keypoint_sequence.append(extract_keypoints(results))

            # Verifica si la secuencia de puntos clave tiene la longitud suficiente
            if len(keypoint_sequence) > MAX_LENGTH_FRAMES and there_hand(results):
                count_frame += 1
            else:
                if count_frame >= MIN_LENGTH_FRAMES:
                    # Realiza la predicción del modelo con la secuencia de puntos clave
                    res = model.predict(np.expand_dims(keypoint_sequence[-MAX_LENGTH_FRAMES:], axis=0))[0]

                    if res[np.argmax(res)] > threshold:
                        sent = actions[np.argmax(res)]
                        sentence.insert(0, sent)  # Inserta la acción detectada en la oración
                        # text_to_speech(sent)  # Convierte la acción a habla (comentado)
                        sentence, repe_sent = format_letter(sent, sentence, repe_sent)  # Formatea la oración

                    count_frame = 0
                    keypoint_sequence = []

            # Dibuja la interfaz gráfica
            cv2.rectangle(image, (0, 0), (640, 35), (245, 117, 16), -1)
            cv2.putText(image, ' | '.join(sentence), FONT_POS, FONT, FONT_SIZE, (255, 255, 255))

            draw_keypoints(image, results)  # Dibuja los puntos clave en la imagen
            cv2.imshow('Sign Language Translator', image)  # Muestra la imagen en una ventana
            if cv2.waitKey(10) & 0xFF == ord('q'):  # Permite salir del bucle presionando 'q'
                break

        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Se carga el modelO
    model_path = os.path.join(DYNAMIC_MODEL_DIR, MODEL_NAME)
    lstm_model = load_model(model_path)  # Carga el modelo entrenado desde el archivo
    evaluate_model(lstm_model)  # Inicia la evaluación del modelo
