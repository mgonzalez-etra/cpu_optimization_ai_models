from fastapi import FastAPI, Request, Response, Body, File, UploadFile
import uvicorn
from starlette.background import BackgroundTask
from fastapi.responses import JSONResponse
from utils.utils import JSONDefaultResponses, load_keras_model, cleanup
from tensorflow import keras
import conversions.tensorflow_to_onnx as tf_onnx
import tempfile
import tensorflow as tf
from protobuf_to_dict import protobuf_to_dict
from fastapi.responses import FileResponse
import base64, json

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
    onnx_model, _ =  tf_onnx.fromKeras(keras_model)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        serialized_model = onnx_model.SerializeToString()
        tmp.write(serialized_model)
        return FileResponse(tmp.name,  filename = str(file.filename).replace('.h5', '.onnx'), background=BackgroundTask(cleanup, tmp.name)) # str solo para forzar el typing, que como es optional mypy llora
    
@app.post("/convert/tensorflow/openvino")
async def tf_to_openvino(file: UploadFile = File(...), type: str = "Keras"):
    model_file = await file.read()
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(model_file)
        model = tf.keras.models.load_model(tmp.name)
        print(model.summary())
        onnx_model, _ =  tf_onnx.fromKeras(model)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        serialized_model = onnx_model.SerializeToString()
        tmp.write(serialized_model)
        response = FileResponse(tmp.name,  filename = str(file.filename).replace('.h5', '.onnx')) # str solo para forzar el typing, que como es optional mypy llora
        return response

@app.post("/upload-file")
async def upload_file(file: UploadFile= File(...)):
    model_file = await file.read()
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(model_file)
        model = tf.keras.models.load_model(tmp.name)
        print(model.summary())
    return file.filename

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)