from tf2onnx import convert
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from fastapi import UploadFile
import tempfile
from utils.utils import cleanup

def fromKeras(model, input_signature=None, opset=None, custom_ops=None,
                custom_op_handlers=None, custom_rewriter=None,
                inputs_as_nchw=None, outputs_as_nchw=None, extra_opset=None,
                shape_override=None, target=None, large_model=False, output_path=None):
    
    model_proto, external_tensor_storage = convert.from_keras(model,
                input_signature, opset, custom_ops,
                custom_op_handlers, custom_rewriter,
                inputs_as_nchw, outputs_as_nchw, extra_opset,
                shape_override, target, large_model, output_path)
    return model_proto, external_tensor_storage

def serialize_onnx_model(onnx_model, file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        serialized_model = onnx_model.SerializeToString()
        tmp.write(serialized_model)
        # str solo para forzar el typing, que como es optional mypy llora
        return FileResponse(tmp.name,  filename=str(file.filename).replace('.h5', '.onnx'), background=BackgroundTask(cleanup, tmp.name))


