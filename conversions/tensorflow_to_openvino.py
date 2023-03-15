from openvino.runtime import compile_model, serialize
from openvino.tools.mo import convert_model
import tensorflow as tf
import tempfile
from fastapi import Response
import os
from fastapi.responses import FileResponse
import zipfile
from starlette.background import BackgroundTask
from utils.utils import cleanup

def tf_to_openvino(model_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(model_file)
        try:
            print(os.path.dirname(__file__))
            model = tf.keras.models.load_model(tmp.name)
            print(model.summary())
            tf.saved_model.save(model,'temp_files/model')
            compiled_model_tf = convert_model(saved_model_dir="temp_files/model", use_new_frontend=True)
            serialize(compiled_model_tf, xml_path="temp_files/openvino_model/serialized.xml", bin_path="temp_files/openvino_model/serialized.bin")
        except Exception as e:
            print(e)
            raise Exception('Model not valid')
        return True
    
def zipfiles(filenames):
    zip_path = 'temp_files/openvino.zip'
    zf = zipfile.ZipFile(zip_path, mode="w")

    for fpath in filenames:
        fdir, fname = os.path.split(fpath)
        zf.write(fpath, fname)

    zf.close()
    return FileResponse(zip_path,  background=BackgroundTask(cleanup, [zip_path, "temp_files/openvino_model", "temp_files/model"]))
