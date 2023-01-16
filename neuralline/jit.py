import os.path
from torch.utils.cpp_extension import load

current_folder = os.path.realpath(os.path.abspath(os.path.dirname(__file__)))

rasterize_cuda = load('rasterize_cuda', [current_folder + '/rasterize_cuda.cpp', current_folder + '/rasterize_cuda_kernel.cu'],
                      verbose=True)
