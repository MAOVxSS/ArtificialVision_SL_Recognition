import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Constantes
LENGTH_KEYPOINTS = 63  # Ajusta según la longitud de los keypoints
NUM_EPOCH = 100  # Número de épocas para entrenar el modelo
MODEL_NAME = "static_model_v3.h5"  # Nombre del archivo del modelo


def get_sequences_and_labels(actions, data_path):
    """
    OBTENER SECUENCIAS Y ETIQUETAS
    Retorna las secuencias de puntos clave (keypoints) y sus etiquetas correspondientes.
    """
    sequences = []
    labels = []

    for label, action in enumerate(actions):
        hdf_path = os.path.join(data_path, f"{action}.h5")
        if not os.path.exists(hdf_path):
            print(f"Advertencia: El archivo {hdf_path} no existe.")
            continue

        data = pd.read_hdf(hdf_path, key='data')

        # Asumiendo que 'data' tiene una columna 'keypoints' que contiene las secuencias de puntos clave
        for i in range(len(data)):
            keypoints = data.iloc[i]['keypoints']
            sequences.append(keypoints)
            labels.append(label)

    if not sequences or not labels:
        print("Error: No se encontraron secuencias o etiquetas válidas.")
        exit(1)

    return sequences, labels


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


def get_model(output_length: int):
    model = Sequential()

    # Primera capa LSTM con 64 unidades, activación ReLU y retorno de secuencias
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(None, LENGTH_KEYPOINTS)))
    model.add(Dropout(0.2))  # Capa de Dropout para evitar el sobreajuste

    # Segunda capa LSTM con 128 unidades, activación ReLU y retorno de secuencias
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))  # Capa de Dropout para evitar el sobreajuste

    # Tercera capa LSTM con 128 unidades y activación ReLU sin retorno de secuencias
    model.add(LSTM(128, return_sequences=False, activation='relu'))
    model.add(Dropout(0.2))  # Capa de Dropout para evitar el sobreajuste

    # Primera capa densa con 64 unidades y activación ReLU
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))  # Capa de Dropout para evitar el sobreajuste

    # Segunda capa densa con 64 unidades y activación ReLU
    model.add(Dense(64, activation='relu'))

    # Tercera capa densa con 32 unidades y activación ReLU
    model.add(Dense(32, activation='relu'))

    # Cuarta capa densa con 32 unidades y activación ReLU
    model.add(Dense(32, activation='relu'))

    # Capa de salida con activación softmax para clasificación multiclase
    model.add(Dense(output_length, activation='softmax'))

    # Compilación del modelo con optimizador Adam y pérdida categórica cruzada
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def training_model(data_path, model_path):
    # Obtiene las acciones (letras) a predecir desde la ruta de datos
    actions = get_actions(data_path)
    if not actions:
        print("Error: No se encontraron acciones válidas en el directorio de datos.")
        exit(1)

    # Obtiene las secuencias y las etiquetas correspondientes
    sequences, labels = get_sequences_and_labels(actions, data_path)

    # Convertir listas de keypoints en arrays de numpy
    sequences = np.array(sequences)

    # Añadir una dimensión de tiempo a los keypoints
    X = np.expand_dims(sequences, axis=1)

    # Convierte las etiquetas en arrays de numpy
    y = to_categorical(labels, num_classes=len(actions)).astype(int)

    # Verificación de las dimensiones de los datos
    print(f"Dimensiones de X: {X.shape}")
    print(f"Dimensiones de y: {y.shape}")

    # Obtiene el modelo con la cantidad de acciones como salida
    model = get_model(len(actions))

    # Definición de callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

    # Entrena el modelo
    model.fit(X, y, epochs=NUM_EPOCH, validation_split=0.2, callbacks=[early_stopping, reduce_lr])

    # Muestra un resumen del modelo
    model.summary()

    # Guarda el modelo entrenado
    model.save(model_path)


if __name__ == "__main__":
    # Define las rutas para los datos y el modelo
    root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    data_path = os.path.abspath(os.path.join(root, "..", "..", "Static_Data", "Data_H5"))
    save_path = os.path.join(root, "Model", "Static_Model_v3")
    model_path = os.path.join(save_path, MODEL_NAME)

    # Verifica si la ruta de los datos existe
    if not os.path.exists(data_path):
        print(f"Error: the path {data_path} does not exist.")
        exit(1)

    # Crea la ruta para guardar el modelo si no existe
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Llama a la función de entrenamiento del modelo
    training_model(data_path, model_path)
