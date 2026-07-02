import cv2
import numpy as np

from rknn.api import RKNN
from pathlib import Path

ONNX_MODEL_PATH = 'models/mobilenetv2_features.onnx'
RKNN_MODEL_PATH = 'models/mobilenetv2_features.rknn'
IMG_PATH = 'testing.jpg'
DATASET_PATH = Path.cwd() / "img_dataset.txt"
QUANTIZE_ON = True
IMG_SIZE = 224
NPU_TARGET = 'rk3588'

def convert_to_rknn():
    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Pre-process config
    # "-I-" stands for Info as a message
    print('-I- Configuring model...')
    rknn.config(
        mean_values=[[123.675, 116.28, 103.53]],    # ImageNet mean (RGB) * 255
        std_values=[[58.395, 57.12, 57.375]],       # ImageNet std (RGB) * 255
        target_platform=NPU_TARGET,
        quantized_algorithm='normal',
        quantized_method = 'channel'
    )
    print("-I- Model configuration successfully.")


    # Load ONNX model
    print('-I- Loading model...')
    ret = rknn.load_onnx(model=ONNX_MODEL_PATH)
    if ret != 0:
        raise RuntimeError(f'-E- Failed to load ONNX model: {ret}')

    print(f'-I- ONNX model loaded successfully.')


    # Build RKNN model for NPU
    print('-I- Building model...')
    ret = rknn.build(
        do_quantization=QUANTIZE_ON, 
        dataset=DATASET_PATH
    )
    if ret != 0:
        raise RuntimeError(f'-E- Failed to build RKNN model: {ret}')

    print('-I- RKNN model built successfully')


    # Export RKNN model
    print('-I- Exporting RKNN model...')
    ret = rknn.export_rknn(RKNN_MODEL_PATH)
    if ret != 0:
        raise RuntimeError(f'-E- Failed to export RKNN model: {ret}')

    print(f'-I- RKNN model has been exported successfully to: {RKNN_MODEL_PATH.resolve()}')


    # Initialize the runtime environment
    print('-I- Initialize the runtime environment...')
    if NPU_TARGET:
        print(f"Initializing RKNN runtime with target: {NPU_TARGET}")
        
        ret = rknn.init_runtime(
            target = NPU_TARGET,
            perf_debug = True,  # Collect performance/latency info
            eval_mem = True     # Collect memory usage info
        )
    else:
        print("Initializing RKNN runtime in normal simulator mode")
        ret = rknn.init_runtime()
    
    if ret != 0:
        raise RuntimeError(f'-E- Failed to initialize the runtime environment: {ret}')
        # exit(ret)
    print('-I- Runtime environment initialized successfully.')

    print("-I- Evaluating model performance...")
    rknn.eval_perf()
    print("-I- Performance evaluation completed.")

    print("-I- Evaluating memory usage...")
    rknn.eval_mem()
    print("-I- Memory evaluation completed.")

    sdk_version = rknn.get_sdk_version()
    print(f"SDK Version: {sdk_version   }")

    # Set inputs for model inference
    input_img = cv2.imread('IMG_PATH')
    input_img = cv2.resize(input_img, (IMG_SIZE, IMG_SIZE))
    input_img = np.expand_dims(input_img, 0)

    # Perform RKNN model inference 
    print('-I- Running RKNN model inference...')
    # input_img = np.random.randn(1, 256, 128, 3).astype(np.float32) # For testing RKNN model inference by creating random input image tensor
    outputs = rknn.inference(inputs=[input_img])
    print("Outputs Shape: ", outputs.shape)

    # Quantitative accuracy analysis
    print('-I- Analysing the accuracy .....')
    rknn.accuracy_analysis(
        inputs=[input_img],
        output_dir = Path.cwd() / "quant_acc_analysis" ,
        target = NPU_TARGET
    )
    print('-I- Quantitative accuracy analysis completed.')

    rknn.release()


if __name__ == '__main__':
    convert_to_rknn()