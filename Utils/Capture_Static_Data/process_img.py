import cv2
import os
from Utils.image_utils import binarize_img
from Constants.constants import STATIC_DATA_WITH_OUT_P_DIR, STATIC_DATA_PROCESSED_DIR

# Carpeta con las imágenes capturadas sin modificaciones
dir = os.path.join(STATIC_DATA_WITH_OUT_P_DIR, "a")  # Cambiar la letra
# Carpeta para las imágenes preprocesadas (binarizadas)
preprocessed_dir = os.path.join(STATIC_DATA_PROCESSED_DIR, "a")  # Cambiar la letra

# Verificar si la carpeta para las imágenes preprocesadas existe, y si no, crearla
if not os.path.exists(preprocessed_dir):
    os.makedirs(preprocessed_dir)

# Procesar cada imagen en la carpeta de imágenes capturadas
for filename in os.listdir(dir):
    if filename.endswith(".jpg"):  # Procesar solo archivos con extensión .jpg
        # Leer la imagen original (RGB)
        img = cv2.imread(os.path.join(dir, filename))

        # Convertir la imagen de RGB a HSV
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Binarizar la imagen
        bin_image = binarize_img(img, hsv_img)

        # Guardar la imagen binarizada en la carpeta de imágenes preprocesadas
        cv2.imwrite(os.path.join(preprocessed_dir, filename), bin_image)

# Cerrar todas las ventanas de OpenCV
cv2.destroyAllWindows()
