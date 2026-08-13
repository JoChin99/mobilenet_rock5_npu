import os
import cv2
import csv
import time
import torch
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

from rknn.api import RKNN
from rknnlite.api import RKNNLite
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay

onnx_model_path = 'models/mobilenetv2_features.onnx'
rknn_model_path = 'models/mobilenetv2_features.rknn'
classifier_path = 'models/classifier.pt'
img_dir = Path('data/2d-geometric-shapes-17-shapes/2D_Geometric_Shapes_Dataset/')
img_size = (224, 224)
npu_target = 'rk3588'
classes = ['circle', 'heart', 'star']
labels = {'circle': 0, 'heart': 1, 'star': 2}


def preprocess_img_onnx(img_path):
    # Float32 NCHW with ImageNet normalization
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    return img


def preprocess_img_rknn(img_path):
    # Raw uint8 NHWC
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0)
    
    return img


def load_classifier(path):
    weights = torch.load(path, weights_only=False)

    classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(1280, 3)
    )
    with torch.no_grad():
        classifier[1].weight.data.copy_(weights["1.weight"])
        classifier[1].bias.data.copy_(weights["1.bias"])
    classifier.eval()
    
    return classifier


def load_imgs():
    img_files = [
        (p, labels[p.parent.name])
        for class_name in classes
        for p in (img_dir / class_name).glob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    return img_files


def evaluate(model_name, img_files, classifier, infer, preprocess_img):
    # Create for Confusion Matrix
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    infer_times = []

    for img_path, label in tqdm(img_files, desc=model_name):
        input_data = preprocess_img(img_path)

        # Measure inference time
        start = time.perf_counter()
        features = infer(input_data)
        infer_times.append((time.perf_counter() - start) * 1000.0)

        with torch.no_grad():
            logits = classifier(torch.from_numpy(features))
            pred = torch.argmax(logits, dim=1).item()

        confusion[label][pred] += 1

    # Compute metrics from confusion matrix
    tp = np.diag(confusion)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp

    times = np.array(infer_times)
    n = len(img_files)

    result = {
        'name': model_name,
        'accuracy': tp.sum() / n,
        'recall': tp / (tp + fn),
        'precision': tp / (tp + fp),
        'confusion': confusion,
        'mean_ms': times.mean(),
        'min_ms': times.min(),
        'max_ms': times.max(),
        'std_ms': times.std(),
        'fps': 1000.0 / times.mean(),
    }
    return result

# Run CPU
def run_cpu_onnx(img_files, classifier):
    session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print("ONNX provider:", session.get_providers())

    def infer(input_data):
        output = session.run(None, {input_name: input_data})[0]
        if output.ndim == 4:
            output = output.mean(axis=(2,3))

        return output

    cpu = evaluate("CPU (fp32)", img_files, classifier, infer, preprocess_img_onnx)

    return cpu

# Run NPU
def run_npu_rknn(img_files, classifier):
    #rknn = RKNN(verbose=False)
    rknn = RKNNLite(verbose=False)
    if rknn.load_rknn(rknn_model_path) != 0:
        raise RuntimeError('-E- Failed to load RKNN model')
    if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
        raise RuntimeError('-E- Failed to initialize RKNN runtime')

    def infer(input_data):
        output = rknn.inference(inputs=[input_data])[0]
        if output.ndim == 4:
            output = output.mean(axis=(2,3))

        return output

    npu = evaluate("NPU (int8)", img_files, classifier, infer, preprocess_img_rknn)
    rknn.release()

    return npu

# Plot confusion matrix
def confMat_plt(cpu, npu):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, result in zip(axes, (cpu, npu)):
        disp = ConfusionMatrixDisplay(confusion_matrix=result['confusion'], display_labels=classes)
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(result['name'], fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

# Compare CPU and NPU (Benchmark)
def comp_model(cpu, npu):
    # To convert bytes to megabytes
    onnx_model_size = os.path.getsize(onnx_model_path) / 1e6
    rknn_model_size = os.path.getsize(rknn_model_path) / 1e6

    print("----- Model Comparison Results -----")
    print(f"{'Metric':<28}{'CPU (ONNX)':>16}{'NPU (RKNN)':>16}")
    print(f"{'Accuracy':<28}{cpu['accuracy']:>16.4f}{npu['accuracy']:>16.4f}")
    for i, name in enumerate(classes):
        print(f"{'Recall (' + name + ')':<28}{cpu['recall'][i]:>16.4f}{npu['recall'][i]:>16.4f}")
    for i, name in enumerate(classes):
        print(f"{'Precision (' + name + ')':<28}{cpu['precision'][i]:>16.4f}{npu['precision'][i]:>16.4f}")
    print(f"{'Mean Latency (ms)':<28}{cpu['mean_ms']:>16.2f}{npu['mean_ms']:>16.2f}")
    print(f"{'Min Latency (ms)':<28}{cpu['min_ms']:>16.2f}{npu['min_ms']:>16.2f}")
    print(f"{'Max Latency (ms)':<28}{cpu['max_ms']:>16.2f}{npu['max_ms']:>16.2f}")
    print(f"{'FPS (img/s)':<28}{cpu['fps']:>16.2f}{npu['fps']:>16.2f}")
    print(f"{'Model Size (MB)':<28}{onnx_model_size:>16.2f}{rknn_model_size:>16.2f}")
    print(f"\nNPU Speedup: {cpu['mean_ms'] / npu['mean_ms']:.2f}x")
    print(f"Accuracy Difference: {npu['accuracy'] - cpu['accuracy']:+.4f}")  #(quantization cost)

    confMat_plt(cpu,npu)


# To save as a log for each run to collect data
def save_csv(cpu, npu):
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    num = len(list(results_dir.glob('cpu_npu_benchmark_*.csv'))) + 1
    csv_path = results_dir / f"cpu_npu_benchmark_{num}.csv"

    onnx_model_size = os.path.getsize(onnx_model_path) / 1e6
    rknn_model_size = os.path.getsize(rknn_model_path) / 1e6

    rows = [
        ['Metric', 'CPU (ONNX)', 'NPU (RKNN)'],
        ['Accuracy', f"{cpu['accuracy']:.4f}", f"{npu['accuracy']:.4f}"],
        ['Mean Latency (ms)', f"{cpu['mean_ms']:.2f}", f"{npu['mean_ms']:.2f}"],
        ['Min Latency (ms)', f"{cpu['min_ms']:.2f}", f"{npu['min_ms']:.2f}"],
        ['Max Latency (ms)', f"{cpu['max_ms']:.2f}", f"{npu['max_ms']:.2f}"],
        ['FPS (img/s)', f"{cpu['fps']:.2f}", f"{npu['fps']:.2f}"],
        ['Model Size (MB)', f"{onnx_model_size:.2f}", f"{rknn_model_size:.2f}"],
        ['NPU Speedup', f"{cpu['mean_ms'] / npu['mean_ms']:.2f}x", ''],
        ['Accuracy Difference', f"{npu['accuracy'] - cpu['accuracy']:+.4f}", '']
    ]
    for i, name in enumerate(classes):
        rows.append([f'Recall ({name})', f"{cpu['recall'][i]:.4f}", f"{npu['recall'][i]:.4f}"])
    for i, name in enumerate(classes):
        rows.append([f'Precision ({name})', f"{cpu['precision'][i]:.4f}", f"{npu['precision'][i]:.4f}"])

    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerows(rows)
        
    print(f"-I- Results saved to: {csv_path}")


if __name__ == '__main__':
    classifier = load_classifier(classifier_path)
    img_files = load_imgs()
    print(f"Total test images found: {len(img_files)}")

    cpu_result = run_cpu_onnx(img_files, classifier)
    npu_result = run_npu_rknn(img_files, classifier)
    comp_model(cpu_result, npu_result)
    save_csv(cpu_result, npu_result) 