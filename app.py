from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from io import BytesIO
import tensorflow as tf
import numpy as np
from PIL import Image

# Inicializar FastAPI
app = FastAPI()

# Cargar el modelo preentrenado
model = tf.keras.applications.MobileNetV2(weights="imagenet")

def procesar_imagen_ml(imagen):
    """
    Procesa una imagen utilizando un modelo preentrenado y devuelve las predicciones principales.
    """
    try:
        # Cargar la imagen desde bytes
        imagen = Image.open(BytesIO(imagen))

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
        return {"estado": "exito", "predicciones": resultado}
    except Exception as e:
        return {"estado": "error", "mensaje": str(e)}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint para procesar una imagen y devolver las predicciones.
    """
    try:
        # Leer contenido del archivo subido
        contents = await file.read()

        # Procesar la imagen
        resultado = procesar_imagen_ml(contents)

        # Devolver el resultado como JSON
        return JSONResponse(content=resultado)
    except Exception as e:
        return JSONResponse(content={"estado": "error", "mensaje": str(e)})
