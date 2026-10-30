# export_onnx.py
import sys
sys.path.insert(0, ".")
import torch
from src.model import ASLNet
from src.dataset import CLASS_NAMES

model = ASLNet(num_classes=len(CLASS_NAMES), width=0.5)
ckpt = torch.load("checkpoints/model_real.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state"])
model.eval()

dummy = torch.randn(1, 3, 128, 128)
torch.onnx.export(
    model, dummy, "asl_net.onnx",
    input_names=["input"],
    output_names=["logits"],
    opset_version=17,
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
)
print("✓ Exported asl_net.onnx")