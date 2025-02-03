#!/bin/bash

source $(conda info --root)/etc/profile.d/conda.sh

conda create -y --name sbalign python=3.9
conda activate sbalign

python -m pip install --upgrade pip
conda env update --file ./sbalign.yml

TORCH=2.4.0
CUDA=cu124

# if [[ "$OSTYPE" == "linux-gnu"* ]]; then
#     CUDA=cu113
# else
#     CUDA=cpu
# fi

ARCH=$(uname -m)

python -m pip install torch_geometric

# if [[ "$ARCH" == "arm64" ]]; then
#     python -m pip install torch_scatter
#     python -m pip install torch_sparse
#     python -m pip install torch_cluster
# else
#     python -m pip install --no-index \
#         torch_scatter \
#         torch_sparse \
#         torch_cluster \
#         -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
# fi

python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html


python -m pip install --upgrade e3nn

# TODO: MAYBE to avoid the setup issues do: python pip install .
python ../setup.py develop
