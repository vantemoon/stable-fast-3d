#!/bin/bash
source $CONDA_DIR/etc/profile.d/conda.sh
conda activate sf3d-env

python /.huggingface/hf_login.py

export USE_CUDA=1
pip install -v ./texture_baker/

exec "$@"
