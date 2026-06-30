import torch
import torch.nn as nn
import torchvision.models as models

from pathlib import Path
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights


# Select device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu") # using CPU only
print(f"Device: {device}")


class MobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()

        # Load model
        mobilenet_model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        self.features = mobilenet_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.features(x)    # [1, 1280, 7, 7]
        x = self.avgpool(x)     # [1, 1280, 1, 1]
        x = nn.flatten(x, 1)    # [1, 1280]
        return x


def export_to_onnx():
    model = MobileNetV2()
    model.eval()

    batch_size = 1
    sample_inputs = torch.randn(batch_size, 3, 224, 224)
    output = model(sample_inputs)
    print("Output shape:", output.shape)

    models_dir = Path.cwd() / "models"
    models_dir.mkdir(exist_ok=True)
    onnx_file = models_dir / "mobilenetv2_features.onnx"

    # Export the model to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            sample_inputs,
            onnx_file,
            #opset_version = 12,
            input_names = ["input"],
            output_names = ["features"],
            dynamo = True,
            dynamic_shapes = None   # Fixed the model to same shapes
        )

    print("Model file are exported here: ", onnx_file.resolve())


if __name__ == "__main__":
    export_to_onnx()

