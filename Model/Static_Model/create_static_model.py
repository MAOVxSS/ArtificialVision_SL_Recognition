import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

class create_static_model:
    @staticmethod
    # Dimensiones de los datos de entrada,  canal de color y clases
    def build_model(input_shape=(224, 224, 1), num_classes=21):
        # Crear una instancia del modelo secuencial
        modelo = Sequential()

        # Primera capa convolucional seguida de una capa de pooling
        modelo.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        modelo.add(MaxPooling2D(pool_size=(2, 2)))

        # Segunda capa convolucional seguida de una capa de pooling
        modelo.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        modelo.add(MaxPooling2D(pool_size=(2, 2)))

        # Tercera capa convolucional sin capa de pooling
        modelo.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

        # Cuarta capa convolucional sin capa de pooling
        modelo.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

        # Quinta capa convolucional seguida de una capa de pooling
        modelo.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
        modelo.add(MaxPooling2D(pool_size=(2, 2)))

        # Aplanado
        modelo.add(Flatten())

        # Capas relu
        modelo.add(Dense(70, activation='relu'))
        modelo.add(Dense(50, activation='relu'))
        modelo.add(Dense(30, activation='relu'))

        # Capa de salida con activación softmax para clasificación multiclase
        modelo.add(Dense(num_classes, activation='softmax'))

        # Compilar el modelo con el optimizador Adam y la pérdida categórica cruzada
        modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Retornar el modelo compilado
        return modelo
