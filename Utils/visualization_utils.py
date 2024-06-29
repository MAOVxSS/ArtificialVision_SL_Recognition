"""
Métodos para la visualización de datos
"""
import pandas as pd
import matplotlib.pyplot as plt

# Este método es para analizar archivos h5, principalmente los que contengan 'keypoints'
def analyze_h5_keypoints(h5_path):
    # Lee el archivo HDF5 con pandas
    df = pd.read_hdf(h5_path, key='data')

    # Muestra la forma del DataFrame y las primeras filas
    print(f"DataFrame shape: {df.shape}")
    print(df.head())

    # Verifica la consistencia de los keypoints
    consistent = df['keypoints'].apply(lambda x: len(x) == 63).all()
    if consistent:
        print("All keypoints entries have 63 elements.")
    else:
        print("Some keypoints entries do not have 63 elements.")

    # Convertir los keypoints en un DataFrame separado para análisis
    keypoints_df = pd.DataFrame(df['keypoints'].tolist())

    # Estadísticas básicas
    print("Basic statistics for keypoints:")
    print(keypoints_df.describe())


# Función para mostrar información extra sobre el entrenamiento de los modelos
def plot_history(history):
    # Graficar la precisión
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Precisión a través de las épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()

    # Graficar la pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida a través de las épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    plt.show()
