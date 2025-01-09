#!/bin/bash
conda create -n contextgs python==3.8.12
conda init
conda activate contextgs
export CUDA_HOME=/usr/local/cuda-11.8

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu116/pyg_lib-0.1.0%2Bpt112cu116-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu116/torch_cluster-1.6.0%2Bpt112cu116-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu116/torch_scatter-2.1.0%2Bpt112cu116-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu116/torch_sparse-0.6.16%2Bpt112cu116-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu116/torch_spline_conv-1.2.1%2Bpt112cu116-cp38-cp38-linux_x86_64.whl
pip install tqdm
pip install einops
pip install tensorboardX
pip install lpips
pip install wandb
pip install laspy
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install opencv-python
pip install colorama
pip install plyfile
pip install jaxtyping
pip install compressai
pip install torchac

conda install -y cudatoolkit=11.6 -c conda-forge
