import cv2
import torch
import numpy as np
import onnxruntime as ort

from tqdm import tqdm
from pathlib import Path

onnx_model_path = 'models/mobilenetv2_features.onnx'
classifier_path = 'models/classifier.pt'
test_results_path = Path.cwd() / 'test/onnx_test_results'
img_dir = Path('data/four-shapes/shapes/')
#img_path = "./test/heart.png"
img_size = (224, 224)
classes = ['circle', 'heart', 'star']
labels = {'circle': 0, 'heart': 1, 'star': 2}

def preprocess_image(img_paths):
    img = cv2.imread(str(img_paths))
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_paths}")
    
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert uint8 to float32
    img = img.astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # (Height, Width, Channels) to (Channels, Height, Width)
    img = np.expand_dims(img, axis=0).astype(np.float32)

    return img

def load_classifier(classifier_path): #, nfeatures=1280, nclasses=3
    # Load classifier
    classifer_model_file = torch.jit.load(classifier_path)
    params = list(classifer_model_file.parameters())
    #classifer_model_file = torch.load(classifier_path)
    classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.2),  
        torch.nn.Linear(1280, 3)
    )
    classifier[1].weight.data.copy_(params[0])
    classifier[1].bias.data.copy_(params[1])
    classifier.eval()

    return classifier

def run_onnx_infer(onnx_model_path, classifier_path, classifier, device_id):
    tp = 0  # true positive
    fp = 0  # false positive
    gtp = 0  # ground truth positives
    true_count = 0  # total correct
    sample_num = 0

    img_files = [
        #p for p in img_dir.rglob("*")
        (p, labels[p.parent.name])
        for class_name in classes
        for p in (img_dir / class_name).glob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    print(f"Total test images found: {len(img_files)} images")

    """
    if'' "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]  
    """
        
    # Initialize the ONNX model
    session = ort.InferenceSession(
        onnx_model_path,
        providers = ["CPUExecutionProvider"],  # CPUExecutionProvider
        provider_options = [{"device_id": device_id}]
    )
    input_name = session.get_inputs()[0].name

    # Check available providers
    print("Available providers:", ort.get_available_providers())
    print("Current provider:", session.get_providers())
    
    for img_path, label in tqdm(img_files):
        sample_num += 1
        
        #classes = {
        #    "circle": 0,
        #    "heart": 1,
        #    "star": 2
        #}
        #label_name = img_path.parent.name
        #label = classes[label_name]

        # Preprocess image
        input_data = preprocess_image(img_path.as_posix())

        # Inference run using image data as the input to the model
        output = session.run(None, {input_name: input_data})
        features = output[0]
        #print("Output shape:", features.shape)

        with torch.no_grad():
            features_tensor = torch.from_numpy(features)
            logits = classifier(features_tensor)
            # score = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1).item()

        if label == pred:
            true_count += 1

        if label == 1:
            gtp += 1
            if pred == 1:
                tp += 1

        if label == 0 and pred == 1:
            fp += 1

    accuracy = true_count / sample_num if sample_num > 0 else 0
    recall = tp / gtp if gtp > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    print(f"Accuracy: {true_count} / {sample_num} = {accuracy:.4f}")
    print(f"Recall: {tp} / {gtp} = {recall:.4f}")
    print(f"Precision: {tp} / {tp + fp} = {precision:.4f}")


if __name__ == "__main__":
    classifier = load_classifier(classifier_path)
    run_onnx_infer(onnx_model_path, classifier_path, classifier, 0)