import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 5)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleModel()
dummy_input = torch.randn(1, 10)
torch.onnx.export(
    model, dummy_input, "simple_model.onnx",
    input_names=["input"], output_names=["output"], opset_version=17
)
print("✅ Model exported successfully to simple_model.onnx")