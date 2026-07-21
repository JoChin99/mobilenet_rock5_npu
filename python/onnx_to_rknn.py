import cv2
import numpy as np

from rknn.api import RKNN
from pathlib import Path
from tqdm import tqdm

onnx_model_path = 'models/mobilenetv2_features.onnx'
rknn_model_path = 'models/mobilenetv2_features.rknn'
img_dir = Path('data/four-shapes/shapes/')
#img_path = 'test/heart.png'
dataset_path = 'img_dataset.txt'
quantize_on = True
img_size = (224,224)
npu_target = 'rk3588'
class_names = ['circle', 'star', 'heart']
labels = {'circle':0, 'star':1, 'heart':2}

def preprocess_image(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img,(2,0,1))
    img = np.expand_dims(img, 0)

    return img

def load_imgs():
    images = []

    for class_name in class_names:
        folder = img_dir / class_name

        for img in folder.glob("*"):
            if img.suffix.lower() in [
                ".png",
                ".jpg",
                ".jpeg"
            ]:
                images.append(
                    (img, labels[class_name])
                )

    return images

def convert_to_rknn():
    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Pre-process config
    # "-I-" stands for Info as a message
    print('-I- Configuring model...')
    rknn.config(
        mean_values=[[123.675, 116.28, 103.53]],    # ImageNet mean (RGB) * 255
        std_values=[[58.395, 57.12, 57.375]],       # ImageNet std (RGB) * 255
        target_platform=npu_target,
        quantized_algorithm='normal',
        quantized_method = 'channel',
    )
    print("-I- Model configuration successfully.")


    # Load ONNX model
    print('-I- Loading model...')
    ret = rknn.load_onnx(model=onnx_model_path)
    if ret != 0:
        raise RuntimeError(f'-E- Failed to load ONNX model: {ret}')

    print(f'-I- ONNX model loaded successfully.')


    # Build RKNN model for NPU
    print('-I- Building model...')
    ret = rknn.build(
        do_quantization=quantize_on, 
        dataset=dataset_path
    )
    if ret != 0:
        raise RuntimeError(f'-E- Failed to build RKNN model: {ret}')

    print('-I- RKNN model built successfully')


    # Export RKNN model
    print('-I- Exporting RKNN model...')
    ret = rknn.export_rknn(rknn_model_path)
    if ret != 0:
        raise RuntimeError(f'-E- Failed to export RKNN model: {ret}')

    print(f'-I- RKNN model has been exported successfully to: {rknn_model_path}')


    # Initialize the runtime environment
    print('-I- Initialize the runtime environment...')
    if npu_target:
        print(f"Initializing RKNN runtime with target: {npu_target}")
        
        ret = rknn.init_runtime(
            target = npu_target,
            perf_debug = True,  # Collect performance/latency info
            eval_mem = True     # Collect memory usage info
        )
    else:
        print("-I- Initializing RKNN runtime in normal simulator mode")
        ret = rknn.init_runtime()
    
    if ret != 0:
        raise RuntimeError(f'-E- Failed to initialize the runtime environment: {ret}')
        # exit(ret)
    print('-I- Runtime environment initialized successfully.')

    print("-I- Evaluating model performance...")
    rknn.eval_perf()
    print("-I- Performance evaluation completed.")

    print("-I- Evaluating memory usage...")
    rknn.eval_memory()
    print("-I- Memory evaluation completed.")

    # Load dataset
    img_data = load_imgs()
    print(f"Total images: {len(img_data)}")

    print('-I- Running RKNN model inference...')
    for img_path, label in tqdm(img_data):
        # Set inputs for model inference
        input_img = preprocess_image(img_path)

        # Perform RKNN model inference 
        # input_img = np.random.randn(1, 256, 128, 3).astype(np.float32) # For testing RKNN model inference by creating random input image tensor
        outputs = rknn.inference(inputs=[input_img])
        #print("Outputs Shape: ", outputs[0].shape)
        #print(img_path.name, "feature shape:", outputs[0].shape)

    # Quantitative accuracy analysis
    image_list = []

    for img in img_dir.rglob("*"):
        if img.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            image_list.append(str(img))

    print('-I- Analysing the accuracy .....')
    rknn.accuracy_analysis(
        inputs=image_list,
        output_dir = './models/quant_acc_analysis',
        target = npu_target
    )
    print('-I- Quantitative accuracy analysis completed.')

    rknn.release()


if __name__ == '__main__':
    convert_to_rknn()