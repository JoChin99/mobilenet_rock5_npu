# MobileNetV2 deployment on ROCK 5 NPU for edge computing and transfer learning
## Overview
Data classification using a frozen MobileNetV2 feature extractor running on the RK3588NPU (Rock5B+) with a trainable classifier layer. A single linear layer is trained with libtorch on the CPU.

**NOTE:** It is recommended to run all Python scripts from the top level `mobilenet_rock5` directory after cloning this repository. Running the script from other directories may require modifying the file paths in the scripts accordingly.

This project is work in progress.

---

## Bill of Materials (BOM)
| Component | Quantity |
|-----------|----------|
| Radxa Rock5B+ 8GB RAM | 1 |
| Radxa Rock5B Case | 1 |
| PD Power Supply | 1 |
| M2 Memory Card | 1 |
| Raspberry Pi Camera Module V2 | 1 |
| M2 Memory Card Reader | 1 |
| USB WiFi Dongle | 1 |

--- 

## Prerequisites
### Environment setup
Run this shell script to create the virtual environment:
```bash
./setup_env.sh
```
Activate the environment:
```bash
source rock5b_env/bin/activate
```

Or run this to create manually:
```bash
python3 -m venv rock5b_env
source rock5b_env/bin/activate
pip install -r requirements.txt
```

### C++ build to compile the transfer learning code
```bash
mkdir build && cd build
cmake ..
make
```

### Install dataset from Kaggle
Run this command:
```bash
python ./python/get_dataset.py
```
To use a different Kaggle dataset, update the `kaggle_dataset` variable witht he corresponding dataset link. The script will then download and process the specified dataset when executed.

---

## Export MobilenetV2 features to ONNX
```bash
python ./python/mobilenetv2_to_onnx.py
```
This script generates an ONNX model file in the models directory, **`models/mobilenetv2_features.onnx`**

---

## Convert ONNX to RKNN (with INT8 quantization)
```bash
python ./python/onnx_to_rknn.py
```
This uses **img_dataset.txt** (list of image paths) as the calibration dataset. It generates a RKNN model file in the models directory, **`models/mobilenetv2_features.rknn`**

---

## Calibration
### Generate Calibration Dataset
Before running the script, generate a calibration dataset by randomly selecting 3,000 (or any desired number of) images from the dataset. The following command is only **provided as an example**:
```bash
ls ~/mobilenet_rock5/data/four-shapes/shapes/{circle,heart,star}/*.png | shuf | head -3000 > img_dataset.txt 
```
Modify the dataset path, class names and the number of images as needed for your own dataset.

### Perform Calibration
Each script targets a different inference backend. Choose any of the following scripts to verify the model, depending on the runtime you wish to use:
```bash
python ./test/onnx_model_test.py
python ./test/rknn_model_test.py
python ./test/rknnlite_model_test.py
python ./test/rknnlite_model_unit_test.py
```
The following table provides an overview of the test scripts:
| Test | Description |
|------|-------------|
| onnx_model_test.py | Batch tests on the ONNX model with the classifier head |
| rknn_model_test.py | Batch tests on the RKNN model via rknn-toolkit2 |
| rknnlite_model_test.py | Batch tests on the RKNN model via rknn-toolkit-lite2 on the target device |
| rknnlite_model_unit_test.py | Single image inference via rknn-toolkit-lite2 |

---

## Transfer learning 
Run to learn to train the final classifier:
```bash
./transfer
```
This script generates an classifier file in the models directory, **`models/classifier.pt`** and logs the loss to **`loss.dat`**

---

## Documentation
### Project Structure 
```text 
mobilenet_rock5/ 
|── data/                       # Dataset used for training and evaluation 
|── model/                      # Trained models and converted RKNN models generated from the script
|── scripts/                    # Training, conversion, and inference scripts 
│   |── export_onnx.py          # Export MobileNetV2 feature extractor to ONNX 
│   |── onnx_to_rknn.py         # Convert ONNX model to RKNN format 
|── test/                       # Calibration and model verification scripts 
│   |── onnx_model_test.py 
│   |── rknn_model_test.py 
│   |── rknnlite_model_test.py 
│   |── rknnlite_model_unit_test.py 
|── CMakeLists.txt
|── README.md  
|── img_datset.txt              # List of image paths
|── mobilenetv2_features.h
|── requirements.txt            # Python package dependencies 
|── setup.sh                    # Script to create a virtual environment 
|── transfer.cpp                # Transfer learning
``` 

---

## License
