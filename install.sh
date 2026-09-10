#!/usr/bin/bash

echo "Creating a virtual environment"
python3 -m venv ./venv

source ./venv/bin/activate

echo "Installig dependencies"
python -m pip install -U pip
python -m pip install -U setuptools
pip install -r ./requirements.txt


python -m ipykernel install --user --name uplift_pipeline

echo "Creating venv.tar.gz"
venv-pack -o ./venv.tar.gz

echo "Setup complete"
