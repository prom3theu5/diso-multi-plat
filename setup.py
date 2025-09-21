import glob
import os
import sys

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)


def get_extensions():
    """Multi-platform extension builder."""
    
    # Base files for all platforms
    main_file = [os.path.join("src", "pybind.cpp")]
    
    # Multi-platform abstraction files
    abstraction_files = [
        os.path.join("src", "device_abstraction.cpp"),
        os.path.join("src", "cpu/cpu_backend.cpp"),
        os.path.join("src", "mps/mps_backend.cpp"),
        os.path.join("src", "mps/mps_tables.cpp"),
    ]
    
    # CUDA-specific files
    source_cuda = glob.glob(os.path.join("src", "cuda/*.cu"))
    
    # Start with base sources
    sources = main_file + abstraction_files
    extension = CppExtension
    
    define_macros = []
    extra_compile_args = {"cxx": ["-O3", "-std=c++17"]}
    
    # Check for CUDA availability
    has_cuda = (torch.cuda.is_available() and (CUDA_HOME is not None)) or os.getenv("FORCE_CUDA", "0") == "1"
    
    if has_cuda:
        print("Building with CUDA support")
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        
        nvcc_flags = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags == "":
            nvcc_flags = ["-O3", "--expt-relaxed-constexpr"]
        else:
            nvcc_flags = nvcc_flags.split(" ")
        
        extra_compile_args["nvcc"] = nvcc_flags
    else:
        print("Building without CUDA support (CPU and MPS only)")
    
    # Check for MPS availability (macOS with Apple Silicon)
    has_mps = sys.platform == "darwin" and hasattr(torch.backends, 'mps')
    if has_mps:
        print("MPS support detected")
        define_macros += [("WITH_MPS", None)]
    
    sources = [s for s in sources]
    include_dirs = ["src", "src/cpu", "src/mps", "src/cuda"]
    
    print("Sources:", sources)
    print("Define macros:", define_macros)
    
    ext_modules = [
        extension(
            "diso._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules


setup(
    name="diso",
    version="0.2.0",  # Version bump for multi-platform support
    author_email="xiwei@ucsd.edu",
    keywords="differentiable iso-surface extraction multi-platform CUDA MPS",
    description="Multi-Platform Differentiable Iso-Surface Extraction Package",
    long_description="""
    DISO (Differentiable Iso-Surface Extraction) now supports multiple platforms:
    - CUDA GPUs (original support)
    - Apple Silicon MPS (new)
    - CPU fallback (new)
    
    The library automatically detects the available hardware and uses the appropriate backend.
    """,
    classifiers=[
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows", 
        "Operating System :: MacOS",  # Added macOS support
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Other Audience",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Framework :: Robot Framework :: Tool",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",  # Added Python 3.12 support
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    license="CC BY-NC 4.0",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.6",
    install_requires=["trimesh", "torch>=1.12.0"],  # Minimum torch version for MPS support
    ext_modules=get_extensions(),
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
    },
    zip_safe=False
)