from openvino.runtime import compile_model

def tf_to_openvino(path):
    return compile_model(path)