import cv2
import torch
import numpy as np

from rknn.api import RKNN
from pathlib import Path
from tqdm import tqdm

rknn_model_path = 'models/mobilenetv2_features.rknn'
classifier_path = 'models/classifier.pt'
test_results_path = Path.cwd() / 'test/rknn_test_results'
img_path = "test/cat.jpg"
img_size = (224, 224)

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def run_rknn_infer(rknn):
    tp = 0  # true positive
    fp = 0  # false positive
    gtp = 0  # ground truth positives
    true_count = 0  # total correct
    sample_num = 0

    img_dir = test_results_path
    img_files = [
        p for p in img_dir.rglob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    print(f"Total test images found: {len(img_files)} images")

    # Load classifier
    classifier = torch.jit.load(classifier_path)
    classifier.eval()

    for img_path in tqdm(img_files):
        sample_num += 1

        label = int(img_path.parent.parent.name)

        input_data = preprocess_image(img_path.as_posix())
        output = rknn.inference(inputs=[input_data])
        features = output[0]                        # shape: (1, 1280)
        memory_detail = rknn.eval_memory()
        print(memory_detail)
        print("Output shape:", features.shape)

        with torch.no_grad():
            features_tensor = torch.from_numpy(output[0])
            logits = classifier(features_tensor)
            # score = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1).item()

        if label == pred:
            true_count += 1

        if label == 1:
            gt += 1
            if pred == 1:
                tp += 1

        if label == 0 and pred == 1:
            fp += 1

    rknn.release()

    accuracy = true_count / sample_num if sample_num > 0 else 0
    recall = tp / gtp if gtp > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    print(f"Accuracy: {true_count} / {sample_num} = {accuracy:.4f}")
    print(f"Recall: {tp} / {gtp} = {recall:.4f}")
    print(f"Precision: {tp} / {tp + fp} = {precision:.4f}")


if __name__ == '__main__':
    # Create RKNN object
    rknn = RKNN(verbose=True)
    ret = rknn.load_rknn(rknn_model_path)

    if ret != 0:
        raise RuntimeError(f'-E- Failed to load RKNN model: {ret}')
    
    ret = rknn.init_runtime('rk3588', eval_mem=True)
    if ret != 0:
        raise RuntimeError(f'-E- Failed to initialize runtime environment: {ret}')

    run_rknn_infer(rknn_model_path, classifier_path)
    

    