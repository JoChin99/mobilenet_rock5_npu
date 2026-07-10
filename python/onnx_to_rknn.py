import cv2
import numpy as np

from rknn.api import RKNN
from pathlib import Path

onnx_model_path = 'models/mobilenetv2_features.onnx'
rknn_model_path = 'models/mobilenetv2_features.rknn'
img_path = "test/cat.jpg"
dataset_path = Path.cwd() / "img_dataset.txt"
quantize_on = True
img_size = 224
npu_target = 'rk3588'

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
        quantized_method = 'channel'
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

    print(f'-I- RKNN model has been exported successfully to: {rknn_model_path.resolve()}')


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
    rknn.eval_mem()
    print("-I- Memory evaluation completed.")

    sdk_version = rknn.get_sdk_version()
    print(f"SDK Version: {sdk_version   }")

    # Set inputs for model inference
    input_img = cv2.imread('img_path')
    input_img = cv2.resize(input_img, (img_size, img_size))
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
        target = npu_target
    )
    print('-I- Quantitative accuracy analysis completed.')

    rknn.release()


if __name__ == '__main__':
    convert_to_rknn()