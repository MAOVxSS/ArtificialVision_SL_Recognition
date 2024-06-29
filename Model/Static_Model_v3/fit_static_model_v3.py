import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from Constants.constants import STATIC_MODEL_v3_DIR
from Utils.visualization_utils import plot_history
# Configuración
RANDOM_SEED = 42
model_save_path = '../Generated_Models/keypoint_classifier.hdf5'
NUM_CLASSES = 21

# Lectura de datos
X_dataset = np.loadtxt(STATIC_MODEL_v3_DIR, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 3) + 1)))
y_dataset = np.loadtxt(STATIC_MODEL_v3_DIR, delimiter=',', dtype='int32', usecols=(0))
X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

# Construcción del modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 3, )),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

# Callbacks
cp_callback = tf.keras.callbacks.ModelCheckpoint(model_save_path, verbose=1, save_weights_only=False, save_best_only=True)
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

# Compilación del modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(X_train, y_train, epochs=1000, batch_size=128, validation_data=(X_test, y_test), callbacks=[cp_callback, es_callback])

# Evaluación del modelo
val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)
print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_acc}')

# Guardar modelo en formato HDF5
model.save(model_save_path)
print(f'Model saved to {model_save_path}')

# Llamada a la función de graficar
plot_history(model.history)



