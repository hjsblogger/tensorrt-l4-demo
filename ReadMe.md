# TensorRT on L4: 40% Latency Reduction with Simple MLP

<img src="https://github.com/user-attachments/assets/fbe6f8e6-01c1-4949-ba90-0197fd6ff78c" width="1000" height="400" alt="TensorRT demo">

Minimal demo: Optimize PyTorch MLP for L4 GPUs via TensorRT.

## Quick Start
1. `git clone <repo> && cd tensorrt-l4-demo`
2. `pip install -r requirements.txt`
3. `python models/export_onnx.py`
4. `python tensorrt/build_engine.py`
5. `python benchmarks/benchmark.py` → 40%+ drop!

## Performance (L4, Batch=32)
| Setup       | Latency (ms) | Reduction |
|-------------|--------------|-----------|
| PyTorch    | 0.40-0.60   | Baseline |
| TRT FP16   | 0.20-0.30   | 40-50%   |

L4 Tips: Use g2-standard-4; monitor with `nvidia-smi`.