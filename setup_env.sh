#!/bin/bash

set -e

ROCK5B_DIR="rock5b_env"

if [ ! -d "$ROCK5B_DIR" ]; then
    echo "-I- Creating virtual environment in ./$ROCK5B_DIR"
    if [ -x /usr/local/bin/python3.12 ]; then
        /usr/local/bin/python3.12 -m venv "$ROCK5B_DIR"
    elif [  -x /usr/bin/python3 ]; then
        python3 -m venv "$ROCK5B_DIR"
    else
        echo "-E- Python3.12 version not found."
        exit 1
    fi
else
    echo "-I- $ROCK5B_DIR environment already exists."
fi

source $ROCK5B_DIR/bin/activate
pip install --upgrade pip

# Download RKNN Related Repositories
mkdir -p rknpu
if [ ! -d "rknpu/rknn-toolkit2" ]; then
    git clone https://github.com/airockchip/rknn-toolkit2.git --depth 1 rknpu/rknn-toolkit2
else
    echo "-I- rknn-toolkit2 already exists."
fi

# Install RKNN Toolkit Lite2 for Python 3.12
pip install rknpu/rknn-toolkit2/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.2-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

pip install -r requirements.txt

echo "-I- Done"
echo "-I- The environment can be activated with this command: source $ROCK5B_DIR/bin/activate"