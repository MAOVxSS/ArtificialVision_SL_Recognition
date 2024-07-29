**Sistema de Visión Artificial para el Reconocimiento de la Lengua de Señas Mexicana**

Requerimientos:
  Versión de Python = 3.9 https://www.python.org/downloads/release/python-390/
  Para instalar las librerías necesarias solo usar el comando: pip install -r requirements.txt

Uso:

  El funcionamiento del sistema se divide en 2, uno para el reconocimiento de señas estáticas y otro para señas dinámicas.
  Dentro de la carpeta "test" se encuentran los dos scripts, "test_dynamic_model" para las señas dinámicas y "test_static_model_v3" para las señas estáticas. 

  Los modelos se descargan de la siguiente ruta: https://drive.google.com/drive/folders/1ZYIIbw3rv30Pm2gLQaEyf1QPdQqqWnPp

  El archivo "dynamic_model.keras" corresponde al modelo a utilizar para las señas dinámicas.
  El archivo "static_keypoint.keras" corresponde al modelo a utilizar para las señas estáticas.

  Es necesario que al descargar el proyecto y utilizarlo, crear una sub carpeta llamada "Generated_Models" dentro de la carpeta principal llamada "Model" (la cual se       encuentra en la raíz del proyecto). Dentro de esta carpeta creada se guardarán los dos archivos con los modelos descargados. Esto porque ya se tiene configurado       esas rutas o en caso de querer hacer las rutas manualmente se debe de ubicar la variable que almacena la ruta en los respectivos scripts. 

  Una vez se tiene guardados y bien referenciados los modelos, se puede ejecutar el script sin problema y empezar a probarlo. 

Si se quiere tener mas informacion sobre las señas a realizar para probar el sistema, se puede consultar el siguiente recurso:

https://drive.google.com/file/d/1IC1ByYs0VMmhh5yOlsogMQtS2vayQxYk/view?usp=drivesdk

Autores: 

  De Jesus Mejia Claudia Andrea
  Alvarez Hernandez Kevin Joel
  Ortiz Vazquez Miguel Alberto
  





