#!/bin/bash --login

set -e

export ENV_PREFIX=$PWD/env

mkdir -p $ENV_PREFIX

eval "$(conda shell.bash hook)"
conda env create --prefix $ENV_PREFIX --file environment-torch.yml --force
conda activate $ENV_PREFIX

echo "------------------------------"
echo "VERIFYING PYTORCH INSTALLATION"
echo "------------------------------"
python -c "import torch;x=torch.rand(5, 3);print(x);print(torch.cuda.is_available())"
