import tensorflow as tf

# Configuración
model_save_path = '../Generated_Models/keypoint_classifier.hdf5'
tflite_save_path = '../Generated_Models/keypoint_classifier.tflite'

# Cargar el modelo guardado
model = tf.keras.models.load_model(model_save_path)
print(f'Model loaded from {model_save_path}')

# Conversión del modelo a TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guardar el modelo TFLite
with open(tflite_save_path, 'wb') as f:
    f.write(tflite_model)
print(f'TFLite model saved to {tflite_save_path}')
