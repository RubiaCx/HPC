from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gemm_cutlass',
    ext_modules=[
        CUDAExtension(
            name='gemm_cutlass',
            sources=['gemm_cutlass.cu'],
            include_dirs=[
                '/home/chenxi/cutlass/include',  # 包含 cutlass.h 的目录
                '/home/chenxi/cutlass/tools/util/include'  # 包含 util/reference/device/gemm.h 的目录
            ],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': [
                    '-O2', 
                    '-arch=sm_80', 
                    '--expt-relaxed-constexpr'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
