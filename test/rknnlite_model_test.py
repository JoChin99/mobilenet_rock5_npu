import cv2
import numpy as np

from rknnlite.api import RKNNLite
from pathlib import Path
from tqdm import tqdm

rknn_model_path = 'models/mobilenetv2_features.rknn'
img_dir = Path('data/four-shapes/shapes/')
#img_path = "~/mobilenet_rock5/test/heart.png"
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
    #img = np.transpose(img,(2,0,1))
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

def run_rknn_lite():
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

    # Load dataset
    img_data = load_imgs()
    print(f"Total images: {len(img_data)}")

    for img_path, label in tqdm(img_data):
        # Set inputs for model inference
        input_img = preprocess_image(img_path)

        # Perform RKNN model inference 
        print('-I- Running RKNN model inference...')
        outputs = rknn_lite.inference(inputs=[input_img], data_format=['nhwc'])
        print("Outputs Shape: ", outputs[0].shape)
        #print(img_path.name, "feature shape:", outputs[0].shape)

    rknn_lite.release()


if __name__ == '__main__':
    run_rknn_lite()