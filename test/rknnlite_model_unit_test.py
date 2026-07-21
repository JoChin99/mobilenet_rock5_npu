import cv2
import numpy as np

from rknnlite.api import RKNNLite
from pathlib import Path

rknn_model_path = 'models/mobilenetv2_features.rknn'
img_size = (224,224)
npu_target = 'rk3588'

def preprocess_image(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0)

    return img

def run_rknn_lite():
    img_path = "test/heart.png"
    
    # Create RKNN object
    rknn_lite = RKNNLite(verbose=True)

    # Load RKNN model
    print('-I- Loading model...')
    ret = rknn_lite.load_rknn(rknn_model_path)
    if ret != 0:
        raise RuntimeError(f'-E- Failed to load RKNN model: {ret}')

    print(f'-I- RKNN model loaded successfully.')

    # Initialize the runtime environment
    print('-I- Initialize the runtime environment...')
    if npu_target:
        print(f"Initializing RKNN runtime with target: {npu_target}")
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0) # RKNNLite.NPU_CORE_0_1_2

    else:
        print("-I- Initializing RKNN runtime in normal simulator mode")
        ret = rknn_lite.init_runtime()
    
    if ret != 0:
        raise RuntimeError(f'-E- Failed to initialize the runtime environment: {ret}')
        # exit(ret)
    print('-I- Runtime environment initialized successfully.')

    # Set inputs for model inference
    input_img = preprocess_image(img_path)

    # Perform RKNN model inference 
    print('-I- Running RKNN model inference...')
    outputs = rknn_lite.inference(inputs=[input_img], data_format=['nhwc'])
    print("Outputs Shape: ", outputs[0].shape)

    rknn_lite.release()


if __name__ == '__main__':
    run_rknn_lite()