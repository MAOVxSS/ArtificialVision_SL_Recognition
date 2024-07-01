import tensorflow as tf

# Cargar el modelo guardado
model = tf.keras.models.load_model('../Generated_Models/static_keypoint.keras')

# Convertir el modelo a TFLite con optimizaciones
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# Convertir el modelo
tflite_model = converter.convert()

# Guardar el modelo TFLite
tflite_model_path = '../Generated_Models/3_letters.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
print(f"TFLite model saved to {tflite_model_path}")

# Analizar el modelo TFLite
tf.lite.experimental.Analyzer.analyze(model_content=tflite_model)
