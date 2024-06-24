"""
Captura de datos para el entrenamiento.
Se detecta una mano y se comienza a grabar con la tecla 's' y se detiene con 'e'
Cada grabación es para una letra dinámica
Se guarda la grabación en 'frames' dentro de la carpeta seleccionada
"""

from mediapipe.python.solutions.hands import Hands
from Utils.utils import *
from Constants.constants import FONT, FONT_POS, FONT_SIZE, FRAME_ACTIONS_PATH, id_cam


def capture_samples(path, margin_frame=2, min_cant_frames=5):
    # Crea la carpeta donde se guardarán las muestras si no existe
    create_folder(path)

    # Inicializa las variables para contar las muestras y los frames
    cant_sample_exist = len(os.listdir(path))
    count_sample = 0
    count_frame = 0
    frames = []
    recording = False

    # Utiliza el modelo de detección de manos de MediaPipe
    with Hands() as hands_model:
        video = cv2.VideoCapture(id_cam)  # Abre la cámara

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                print("Error: no se pudo leer el frame del video.")
                break

            # Realiza la detección de manos en el frame actual
            image, results = mediapipe_detection(frame, hands_model)

            if recording and there_hand(results):
                print("Mano detectada.")
                count_frame += 1
                if count_frame > margin_frame:
                    # Muestra el texto "Capturando..." en la imagen
                    cv2.putText(image, 'Capturando...', FONT_POS, FONT, FONT_SIZE, (255, 50, 0))
                    frames.append(np.asarray(frame))  # Guarda el frame actual

            elif recording and not there_hand(results) or not recording and frames:
                if len(frames) > min_cant_frames + margin_frame:
                    print(f"Guardando {len(frames)} frames.")
                    frames = frames[:-margin_frame]  # Ignora los frames al final
                    output_folder = os.path.join(path, f"sample_{cant_sample_exist + count_sample + 1}")
                    create_folder(output_folder)
                    save_frames(frames, output_folder)  # Guarda los frames en la carpeta especificada
                    print(f"Frames guardados en: {output_folder}")
                    count_sample += 1

                frames = []
                count_frame = 0

            # Muestra las instrucciones en la pantalla
            draw_text = 'Presiona "e" para detener' if recording else 'Presiona "s" para comenzar, "q" para salir'
            cv2.putText(image, draw_text, FONT_POS, FONT, FONT_SIZE, (0, 255, 0) if recording else (255, 0, 0))

            # Dibuja los puntos clave de la mano en la imagen
            draw_keypoints(image, results)
            cv2.imshow(f'Toma de muestras para "{os.path.basename(path)}"', image)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('s') and not recording:
                # Comienza la grabación
                recording = True
                frames = []
                count_frame = 0
                print("Grabación iniciada...")
            elif key == ord('e') and recording:
                # Detiene la grabación
                recording = False
                print("Grabación detenida.")
            elif key == ord('q'):
                # Sale del bucle y cierra el programa
                break

        # Libera la cámara y destruye todas las ventanas de OpenCV
        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    word_name = "z" # Cambiar a la letra que se quiera guardar
    word_path = os.path.join(FRAME_ACTIONS_PATH, word_name)  # Ruta de guardado de los frames
    print(f"Guardando frames en: {word_path}")
    capture_samples(word_path)
