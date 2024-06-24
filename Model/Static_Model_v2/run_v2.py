import matplotlib.pyplot as plt
from create_static_model_v2 import create_static_model
from fit_static_model_v2 import fit_static_model
from Constants.constants import STATIC_DATA_WITH_OUT_P_DIR

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


if __name__ == "__main__":
    model = create_static_model()
    history = fit_static_model(model, STATIC_DATA_WITH_OUT_P_DIR)

    # Guardar el modelo entrenado
    model.save('static_sign_language_model.keras', save_format='keras')

    # Mostrar información de la historia de entrenamiento
    print("Entrenamiento completado")
    print("Precisión final (entrenamiento):", history.history['accuracy'][-1])
    print("Precisión final (validación):", history.history['val_accuracy'][-1])
    print("Pérdida final (entrenamiento):", history.history['loss'][-1])
    print("Pérdida final (validación):", history.history['val_loss'][-1])

    # Graficar la historia de entrenamiento
    plot_history(history)
