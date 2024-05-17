#!/bin/bash

PYTHONPATH="$PYTHONPATH:/IdeaProjects/InfoOdometry/flownet/channelnorm_package"
PYTHONPATH="$PYTHONPATH:/IdeaProjects/InfoOdometry/flownet/correlation_package"
PYTHONPATH="$PYTHONPATH:/IdeaProjects/InfoOdometry/flownet/resample2d_package"

CUDA_HOME="/usr/local/cuda-12.1"
LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:/opt/conda/lib/python3.10/site-packages/torch/lib/lib

cd ./flownet/correlation_package
rm -rf *_cuda.egg-info build dist __pycache__
python3 setup.py build_ext --inplace

cd ../resample2d_package
rm -rf *_cuda.egg-info build dist __pycache__
python3 setup.py build_ext --inplace

cd ../channelnorm_package
rm -rf *_cuda.egg-info build dist __pycache__
python3 setup.py build_ext --inplace

cd ..
cd ..