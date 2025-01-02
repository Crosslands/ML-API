
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO

# Cargar el modelo pcdreentrenado
model = tf.keras.applications.MobileNetV2(weights="imagenet")

def procesar_imagen_ml(imagen):
    """
    Procesa una imagen utilizando un modelo preentrenado y devuelve las predicciones principales.
    :param imagen: Bytes o archivo de imagen.
    :return: Lista de predicciones con nombres y probabilidades.
    """
    try:
        # Cargar la imagen desde bytes o archivo
        if isinstance(imagen, bytes):
            imagen = Image.open(BytesIO(imagen))
        elif isinstance(imagen, str):  # Si es una ruta de archivo
            imagen = Image.open(imagen)
        else:
            raise ValueError("Formato de imagen no reconocido.")

        # Redimensionar y normalizar la imagen
        img_resized = imagen.resize((224, 224))
        img_array = np.array(img_resized) / 255.0  # Normalización
        img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        # Realizar la predicción
        predicciones = model.predict(img_array)
        predicciones_decodificadas = tf.keras.applications.mobilenet_v2.decode_predictions(predicciones, top=3)[0]

        # Preparar el resultado
        resultado = [{"nombre": pred[1], "probabilidad": pred[2] * 100} for pred in predicciones_decodificadas]
        return {"estado": "éxito", "predicciones": resultado}
    except Exception as e:
        return {"estado": "error", "mensaje": str(e)}
