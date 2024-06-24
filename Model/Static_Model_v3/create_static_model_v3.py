from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from Constants.constants import LENGTH_KEYPOINTS, MAX_LENGTH_FRAMES

NUM_EPOCH = 100  # Número de épocas para entrenar el modelo

def get_model(output_length: int):
    model = Sequential()

    # Primera capa LSTM con 64 unidades, activación ReLU y retorno de secuencias
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(MAX_LENGTH_FRAMES, LENGTH_KEYPOINTS)))
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
