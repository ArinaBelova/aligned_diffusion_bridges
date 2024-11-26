#!/bin/bash

source $(conda info --root)/etc/profile.d/conda.sh

conda create -y --name sbalign_test python=3.9
conda activate sbalign_test

python -m pip install --upgrade pip
conda env update --file $PWD/env.yml

TORCH=2.4.0
CUDA=cu124

# if [[ "$OSTYPE" == "linux-gnu"* ]]; then
#     CUDA=cu124
# else
#     CUDA=cpu
# fi

ARCH=$(uname -m)

python -m pip install --upgrade --force-reinstall torch_geometric

# if [[ "$ARCH" == "arm64" ]]; then
#     python -m pip install torch_scatter
#     python -m pip install torch_sparse
#     python -m pip install torch_cluster
# else
#     python -m pip install --upgrade --force-reinstall --no-index \
#         torch_scatter \
#         torch_sparse \
#         torch_cluster \
#         -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
# fi

python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

python -m pip install --upgrade e3nn
python setup.py develop
