import os
import numpy as np
from Model.Dynamic_Model.create_dynamic_model import NUM_EPOCH, get_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from Utils.utils import get_actions, get_sequences_and_labels
from Constants.constants import *


# from keras.callbacks import EarlyStopping

def training_model(data_path, model_path):
    # Obtiene las acciones (letras) a predecir desde la ruta de datos
    actions = get_actions(data_path)

    # Obtiene las secuencias y las etiquetas correspondientes
    sequences, labels = get_sequences_and_labels(actions, data_path)

    # Rellena las secuencias para que todas tengan la misma longitud
    sequences = pad_sequences(sequences, maxlen=MAX_LENGTH_FRAMES, padding='post', truncating='post', dtype='float32')

    # Convierte las secuencias y etiquetas en arrays de numpy
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    # Obtiene el modelo con la cantidad de acciones como salida
    model = get_model(len(actions))

    # Definición de parada temprana (comentado)
    # early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    # Entrena el modelo
    model.fit(X, y, epochs=NUM_EPOCH)
    # model.fit(X, y, epochs=NUM_EPOCH, callbacks=[early_stopping])

    # Muestra un resumen del modelo
    model.summary()

    # Guarda el modelo entrenado
    model.save(model_path)


if __name__ == "__main__":
    # Define las rutas para los datos y el modelo
    model_path = os.path.join(DYNAMIC_MODEL_DIR, MODEL_NAME)

    # Verifica si la ruta de los datos existe
    if not os.path.exists(DYNAMIC_DATA_DIR):
        print(f"Error: the path {DYNAMIC_DATA_DIR} does not exist.")
        exit(1)

    # Llama a la función de entrenamiento del modelo
    training_model(DYNAMIC_DATA_DIR, model_path)
