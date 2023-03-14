from tf2onnx import convert

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



