from setuptools import setup, Extension
import sys
import numpy as np

# Adding OpenCV to project
# ************************

# Adding sources of the project
# *****************************

SOURCES = [
    "../cpp_utils/cloud/cloud.cpp",
    "grid_subsampling/grid_subsampling.cpp",
    "wrapper.cpp",
]

# Set platform-appropriate compiler flags
if sys.platform == "win32":
    # For MSVC (VS 2017+), request C++14; older MSVC may ignore this
    extra_compile_args = ["/std:c++14"]
else:
    # GCC/Clang
    extra_compile_args = ["-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"]

module = Extension(
    name="grid_subsampling",
    sources=SOURCES,
    extra_compile_args=extra_compile_args,
)

setup(
    ext_modules=[module],
    include_dirs=[np.get_include()],
)








