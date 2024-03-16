#!/usr/bin/env python3
import os
import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++17']

nvcc_args = [
    '-arch', 'sm_80',
    '-gencode', 'arch=compute_80,code=sm_80',
    '-gencode', 'arch=compute_86,code=sm_86',
    '-gencode', 'arch=compute_87,code=sm_87',
    '-gencode', 'arch=compute_86,code=compute_86'
]

setup(
    name='resample2d_cuda',
    ext_modules=[
        CUDAExtension('resample2d_cuda', [
            'resample2d_cuda.cc',
            'resample2d_kernel.cu'
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
