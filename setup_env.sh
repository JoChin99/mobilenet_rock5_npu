#!/bin/bash

set -e

ROCK5B_DIR="rock5b_env"

if [ ! -d "$ROCK5B_DIR" ]; then
    echo "-I- Creating virtual environment in ./$ROCK5B_DIR"
    python3 -m venv "$ROCK5B_DIR"
else
    echo "-I- $ROCK5B_DIR environment already exists."
fi

source $ROCK5B_DIR/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "-I- Done"
echo "-I- The environment can be activated with this command: source $ROCK5B_DIR/bin/activate"