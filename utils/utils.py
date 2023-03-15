from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
import json
from starlette.background import BackgroundTask
from fastapi import UploadFile
import tempfile
import tensorflow as tf
import os
import shutil


def replace_string_json(json_file, replace):
    if isinstance(json_file, dict):
        for key, value in json_file.items():
            json_file[key] = replace_string_json(value, replace)
    elif isinstance(json_file, list):
        for index, item in enumerate(json_file):
            json_file[index] = replace_string_json(item, replace)
    elif isinstance(json_file, str):
        for index, replace in enumerate(replace):
            json_file = json_file.replace(f"[{index+1}]", replace)
    return json_file


"""
Given the path from the request scope, loads the json with the file structure based on the path itself.
For example, for a call to /help/tf2onnx, will answer with the json from responses/help/tf2onnx.json
"""


def JSONDefaultResponses(api_path, *args):
    json_path = f"responses/{api_path[1:]}.json"
    json_file = json.load(open(json_path, 'r'))
    if args:
        json_file = replace_string_json(json_file, args)
    return JSONResponse(content=json_file)


async def load_keras_model(file: UploadFile):
    model_file = await file.read()
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(model_file)
        model = tf.keras.models.load_model(tmp.name)
    return model


def cleanup(filenames):
    for filename in filenames:
        if os.path.isfile(filename):
            os.remove(filename)
        elif os.path.isdir(filename):
            shutil.rmtree(filename, ignore_errors=True)



