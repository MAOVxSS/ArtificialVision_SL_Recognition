import os
import pandas as pd
from Utils.utils import *

def inspect_hdf5_file(file_path):
    """
    Inspeccionar el contenido de un archivo HDF5 y mostrar su estructura.

    Args:
    file_path (str): Ruta al archivo HDF5.
    """
    if not os.path.exists(file_path):
        print(f"Error: El archivo {file_path} no existe.")
        return

    with pd.HDFStore(file_path, 'r') as hdf:
        # Mostrar todas las claves en el archivo HDF5
        print("Claves en el archivo HDF5:")
        print(hdf.keys())

        # Leer los datos y mostrar la estructura
        data = hdf.get('data')
        print("\nEstructura de los datos:")
        print(data.head())
        print("\nInformación del DataFrame:")
        print(data.info())


if __name__ == "__main__":
    # Ruta al archivo HDF5 que deseas inspeccionar
    file_path = '../Dynamic_Data/j.h5'  # Reemplaza con la ruta real a tu archivo HDF5

    # Inspeccionar el archivo HDF5
    inspect_hdf5_file(file_path)
