import time
import numpy as np
import torch
from models.simple_model import SimpleModel  # Import your class
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# PyTorch Baseline
device = torch.device('cuda')
pytorch_model = SimpleModel().to(device).eval()
pytorch_input = torch.randn(32, 10).to(device)

torch.cuda.synchronize()
for _ in range(100):  # Warmup
    _ = pytorch_model(pytorch_input)
torch.cuda.synchronize()

pytorch_times = []
for _ in range(1000):
    torch.cuda.synchronize()
    start_iter = time.time()
    output = pytorch_model(pytorch_input)
    torch.cuda.synchronize()
    pytorch_times.append(time.time() - start_iter)
pytorch_avg = np.mean(pytorch_times) * 1000
print(f"PyTorch Avg (batch=32): {pytorch_avg:.2f} ms")

# TensorRT (use build_engine from tensorrt/)
from tensorrt.build_engine import build_engine
engine = build_engine("simple_model.onnx")
if engine is None: raise ValueError("Build failed")
context = engine.create_execution_context()
context.set_binding_shape(0, (32, 10))

input_size = trt.volume(context.get_binding_shape(0)) * np.float32(1).itemsize
output_size = trt.volume(context.get_binding_shape(1)) * np.float32(1).itemsize
h_input = np.random.randn(32, 10).astype(np.float32)
h_output = np.empty((32, 5), dtype=np.float32)
d_input = cuda.mem_alloc(input_size)
d_output = cuda.mem_alloc(output_size)
bindings = [int(d_input), int(d_output)]

# Warmup
cuda.memcpy_htod(d_input, h_input)
for _ in range(100):
    context.execute_v2(bindings)
    cuda.memcpy_dtoh(h_output, d_output)

# Time
trt_times = []
for _ in range(1000):
    cuda.memcpy_htod(d_input, h_input)
    start_iter = time.time()
    context.execute_v2(bindings)
    cuda.memcpy_dtoh(h_output, d_output)
    trt_times.append(time.time() - start_iter)
trt_avg = np.mean(trt_times) * 1000
print(f"TensorRT Avg (batch=32, FP16): {trt_avg:.2f} ms")

reduction = (1 - trt_avg / pytorch_avg) * 100
print(f"Reduction: {reduction:.1f}% (Speedup: {pytorch_avg / trt_avg:.2f}x)")

d_input.free()
d_output.free()