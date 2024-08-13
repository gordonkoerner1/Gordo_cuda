from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
import os

# Define the output directory relative to the script's location
output_dir = os.path.join(os.path.dirname(__file__), 'Gordo_cuda')
os.makedirs(output_dir, exist_ok=True)

# Paths to CUDA Toolkit include and lib directories
cuda_include_dir = '/usr/local/cuda/include'  # Adjust based on your CUDA installation
cuda_lib_dir = '/usr/local/cuda/lib64'  # Adjust based on your CUDA installation

extensions = [
    Extension('Gordo_cuda.cusolver_module',
              sources=['cusolver_module.pyx'],
              libraries=['cusolver', 'cuda', 'cudart', 'cusparse'],
              include_dirs=[np.get_include(), cuda_include_dir, '/usr/local/cuda/extras'],
              library_dirs=[cuda_lib_dir]),
    Extension('Gordo_cuda.cusparse_module',
              sources=['cusparse_module.pyx'],
              libraries=['cusparse', 'cuda', 'cudart', 'cusolver'],
              include_dirs=[np.get_include(), cuda_include_dir, '/usr/local/cuda/extras'],
              library_dirs=[cuda_lib_dir])
]

setup(
    name='Gordo_cuda',
    version='0.1',
    packages=find_packages(),
    ext_modules=cythonize(extensions, annotate=True, verbose=True),
    include_dirs=[np.get_include(), cuda_include_dir],
)
