import onnx
import torch
import torch.nn as nn
import torchvision.models as models

from pathlib import Path
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

class MobileNetV2Features(nn.Module):
    def __init__(self):
        super().__init__()

        # Load model
        mobilenet_model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        self.features = mobilenet_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.features(x)    # [1, 1280, 7, 7]
        x = self.avgpool(x)     # [1, 1280, 1, 1]
        x = torch.flatten(x, 1)    # [1, 1280]
        return x


def export_to_onnx():
    model = MobileNetV2Features()
    model.eval()

    batch_size = 1
    sample_inputs = torch.randn(batch_size, 3, 224, 224)
    output = model(sample_inputs)
    print("Output shape:", output.shape)

    models_dir = Path.cwd() / "models"
    models_dir.mkdir(exist_ok=True)
    onnx_path = models_dir / "mobilenetv2_features.onnx"

    # Export the model to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            sample_inputs,
            onnx_path,
            opset_version = 17,
            input_names = ["input"],
            output_names = ["features"]
        )

    print("Model file is exported here: ", onnx_path.resolve())

    return onnx_path

def load_onnx_model():
    onnx_path = export_to_onnx()

    # Load the exported model to verify and check the model
    onnx_model = onnx.load(onnx_path)
    
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print('The model is invalid: %s' % e)
    else:
        print('The model is valid.')
              

if __name__ == "__main__":
    export_to_onnx()
    load_onnx_model()

