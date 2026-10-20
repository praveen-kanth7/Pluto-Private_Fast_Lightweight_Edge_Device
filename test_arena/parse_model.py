import tflite

def get_builtin_code_name(opcode):
    for name in dir(tflite.BuiltinOperator):
        if getattr(tflite.BuiltinOperator, name) == opcode:
            return name
    return "UNKNOWN"

def get_tensor_type_name(t_type):
    for name in dir(tflite.TensorType):
        if getattr(tflite.TensorType, name) == t_type:
            return name
    return "UNKNOWN"

with open("tiny_ds_cnn_int8.tflite", "rb") as f:
    buf = f.read()

print(f"Model file size: {len(buf)} bytes")
model = tflite.Model.GetRootAsModel(buf, 0)
subgraph = model.Subgraphs(0)

print(f"Number of tensors: {subgraph.TensorsLength()}")

inputs = [subgraph.Inputs(i) for i in range(subgraph.InputsLength())]
outputs = [subgraph.Outputs(i) for i in range(subgraph.OutputsLength())]

in_tensor = subgraph.Tensors(inputs[0])
out_tensor = subgraph.Tensors(outputs[0])

in_shape = [in_tensor.Shape(i) for i in range(in_tensor.ShapeLength())]
out_shape = [out_tensor.Shape(i) for i in range(out_tensor.ShapeLength())]

print(f"Input shape: {in_shape}")
print(f"Output shape: {out_shape}")
print(f"Input type: {get_tensor_type_name(in_tensor.Type())}")
print(f"Output type: {get_tensor_type_name(out_tensor.Type())}")

operators = set()
for i in range(model.OperatorCodesLength()):
    opcode = model.OperatorCodes(i)
    operators.add(get_builtin_code_name(opcode.BuiltinCode()))

print("Operators:", ", ".join(operators))

# Calculate parameter count
params = 0
for i in range(subgraph.TensorsLength()):
    tensor = subgraph.Tensors(i)
    buffer = model.Buffers(tensor.Buffer())
    if buffer.DataLength() > 0:
        shape = [tensor.Shape(j) for j in range(tensor.ShapeLength())]
        size = 1
        for s in shape: size *= s
        if size > 16: # skip small scales/zeros
            params += size
            
print(f"Estimated Parameters: {params}")
