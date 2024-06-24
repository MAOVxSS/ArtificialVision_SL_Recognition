import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

class fit_static_model:
    @staticmethod
    def train_model(model, data_dir, batch_size=32, epochs=10, model_path='static_model.keras'):
        # Generador de datos de entrenamiento con aumento de datos y reescalado
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,  # Reescalar los valores de píxeles de [0, 255] a [0, 1]
            shear_range=0.2,  # Aplicar transformaciones de corte
            zoom_range=0.2,  # Aplicar zoom aleatorio
            horizontal_flip=True,  # Voltear las imágenes horizontalmente
            validation_split=0.2  # Utilizar el 20% de los datos para validación
        )

        # Generador de datos de entrenamiento
        train_generator = train_datagen.flow_from_directory(
            data_dir,  # Directorio con las imágenes de entrenamiento
            target_size=(224, 224),  # Redimensionar las imágenes a 224x224 píxeles
            color_mode='grayscale',  # Cargar las imágenes en escala de grises
            batch_size=batch_size,  # Tamaño del lote
            class_mode='categorical',  # Modo de clasificación categórica
            subset='training'  # Subconjunto de entrenamiento
        )

        # Generador de datos de validación
        validation_generator = train_datagen.flow_from_directory(
            data_dir,  # Directorio con las imágenes de validación
            target_size=(224, 224),  # Redimensionar las imágenes a 224x224 píxeles
            color_mode='grayscale',  # Cargar las imágenes en escala de grises
            batch_size=batch_size,  # Tamaño del lote
            class_mode='categorical',  # Modo de clasificación categórica
            subset='validation'  # Subconjunto de validación
        )

        # Checkpoint para guardar el mejor modelo basado en la pérdida de validación
        checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min')
        callbacks = [checkpoint]

        # Entrenar el modelo
        history = model.fit(
            train_generator,  # Generador de datos de entrenamiento
            steps_per_epoch=train_generator.samples // batch_size,  # Número de pasos por época
            validation_steps=validation_generator.samples // batch_size,  # Número de pasos de validación por época
            validation_data=validation_generator,  # Generador de datos de validación
            epochs=epochs,  # Número de épocas para entrenar
            callbacks=callbacks  # Lista de callbacks, incluyendo el checkpoint
        )

        # Retornar el historial del entrenamiento
        return history
