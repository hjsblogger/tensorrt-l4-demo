import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# Assume engine serialized as 'simple_model.engine'
engine = load_engine("simple_model.engine")  # Serialize in build_engine if needed
context = engine.create_execution_context()
context.set_binding_shape(0, (1, 10))

input_size = trt.volume(context.get_binding_shape(0)) * np.float32(1).itemsize
output_size = trt.volume(context.get_binding_shape(1)) * np.float32(1).itemsize
h_input = np.random.randn(1, 10).astype(np.float32)
h_output = np.empty((1, 5), dtype=np.float32)

d_input = cuda.mem_alloc(input_size)
d_output = cuda.mem_alloc(output_size)
bindings = [int(d_input), int(d_output)]

cuda.memcpy_htod(d_input, h_input)
context.execute_v2(bindings)
cuda.memcpy_dtoh(h_output, d_output)

print("Input (first 5):", h_input[0][:5])
print("Output:", h_output[0])
print("Argmax:", np.argmax(h_output[0]))

d_input.free()
d_output.free()