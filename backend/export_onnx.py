from pathlib import Path

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torchvision.models as tv_models


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


class MobileNetModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.mobilenet = tv_models.mobilenet_v2(weights=None)
        in_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.mobilenet(x)


def export_mobilenet(source_name, target_name):
    model = MobileNetModel(2)
    state = torch.load(MODEL_DIR / source_name, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        MODEL_DIR / target_name,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )


def export_segmentation():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=1,
        activation=None,
    )
    state = torch.load(MODEL_DIR / "best_model.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.zeros(1, 1, 224, 224, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        MODEL_DIR / "best_model.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not (MODEL_DIR / "mobilenet.onnx").exists():
        export_mobilenet("mobilenet.pt", "mobilenet.onnx")

    if not (MODEL_DIR / "mobilenet_irrelevent.onnx").exists():
        export_mobilenet("mobilenet_irrelevent.pt", "mobilenet_irrelevent.onnx")

    if not (MODEL_DIR / "best_model.onnx").exists():
        export_segmentation()

    required = [
        MODEL_DIR / "mobilenet.onnx",
        MODEL_DIR / "mobilenet_irrelevent.onnx",
        MODEL_DIR / "best_model.onnx",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"ONNX conversion did not create: {missing}")

    print("ONNX conversion completed successfully.")


if __name__ == "__main__":
    main()
