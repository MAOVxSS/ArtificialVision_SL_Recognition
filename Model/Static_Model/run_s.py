from Model.Static_Model.create_static_model import *
from Model.Static_Model.fit_static_model import *
from Constants.constants import static_data_processed_dir
from Utils.visualization_utils import plot_history

if __name__ == "__main__":

    # Crear la arquitectura del modelo
    model = create_static_model.build_model()

    # Entrenar el modelo
    history = fit_static_model.train_model(model, static_data_processed_dir)

    # Guardar el modelo entrenado
    model.save('final_static_model.keras', save_format='keras')

    # Mostrar información de la historia de entrenamiento
    print("Entrenamiento completado")
    print("Precisión final (entrenamiento):", history.history['accuracy'][-1])
    print("Precisión final (validación):", history.history['val_accuracy'][-1])
    print("Pérdida final (entrenamiento):", history.history['loss'][-1])
    print("Pérdida final (validación):", history.history['val_loss'][-1])

    # Graficar la historia de entrenamiento
    plot_history(history)
