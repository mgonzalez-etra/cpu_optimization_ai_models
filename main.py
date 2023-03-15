from fastapi import FastAPI, Request, Response, Body, File, UploadFile
import uvicorn
from starlette.background import BackgroundTask
from fastapi.responses import JSONResponse
from utils.utils import JSONDefaultResponses, load_keras_model, cleanup
import conversions.tensorflow_to_openvino as tf_opnvn
from tensorflow import keras
import conversions.tensorflow_to_onnx as tf_onnx
from conversions.tensorflow_to_onnx import serialize_onnx_model
import tempfile
import tensorflow as tf
from protobuf_to_dict import protobuf_to_dict
from fastapi.responses import FileResponse
import base64
import json

app = FastAPI()


@app.get("/help")
async def help(request: Request):
    print(request.scope)
    return JSONDefaultResponses(f"{request.scope['path']}/help")


@app.get("/help/{conversion}")
async def helptf2onnx(request: Request, conversion):
    print(request.scope)
    try:
        return JSONDefaultResponses(request.scope['path'])
    except FileNotFoundError:
        return JSONDefaultResponses("/errors/FileNotFoundError", conversion)


@app.post("/convert/tensorflow/onnx")
async def tf_to_onnx(file: UploadFile = File(...), type: str = "Keras"):
    keras_model = await load_keras_model(file)
    onnx_model, _ = tf_onnx.fromKeras(keras_model)
    return serialize_onnx_model(onnx_model, file)


@app.post("/convert/tensorflow/openvino")
async def tf_to_openvino(file: UploadFile = File(...), type: str = "Keras"):
    model_file = await file.read()
    tf_opnvn.tf_to_openvino(model_file) # Guarda el xml y el bin en /temp_files/openvino_model
    res_filename = f"{str(file.filename).replace('.h5', '')}_openvino"
    return tf_opnvn.zipfiles(['temp_files/openvino_model/serialized.xml', 'temp_files/openvino_model/serialized.bin'])
    



@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    model_file = await file.read()
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(model_file)
        model = tf.keras.models.load_model(tmp.name)
        print(model.summary())
    res_filename = f"{str(file.filename).replace('.h5', '')}_openvino."
    opnvn_xml = FileResponse('/temp_files/openvino_model/serialized.xml', filename=f"{res_filename}.xml")
    opnvn_bin = FileResponse('/temp_files/openvino_model/serialized.bin', filename=f"{res_filename}.bin")
    return opnvn_xml, opnvn_bin

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
