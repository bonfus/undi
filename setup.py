from setuptools import setup
import os
from glob import glob

use_blas = False

MKL_L = os.getenv("MKL_LIB", "-L/opt/intel/oneapi/mkl/latest/lib/intel64")
# For GNU:
MKL_L += " -lmkl_gf_lp64  -lmkl_gnu_thread -lmkl_core -D__MKL"
# For Intel:
# MKL_L += " -lmkl_intel_lp64  -lmkl_intel_thread -lmkl_core -D__MKL"
MKL_I = os.getenv("MKL_INCLUDE", "/opt/intel/oneapi/mkl/latest/include/")

# Pybind11
try:
    from pybind11.setup_helpers import build_ext, Pybind11Extension
    ext_modules = [
        Pybind11Extension(
            "fast_quantum",
            ["undi/fast/permutations.cpp", "undi/fast/fast.cpp"],
            extra_compile_args='-O3 -fopenmp -ffast-math -g'.split(),
            extra_link_args='-O3 -fopenmp -ffast-math -g'.split(),
        ),
        Pybind11Extension(
            "fast_quantum_light",
            ["undi/fast/permutations.cpp", "undi/fast/fast_light.cpp"],
            extra_compile_args='-O3 -fopenmp -ffast-math -g'.split(),
            extra_link_args='-O3 -fopenmp -ffast-math -g'.split(),
        ),
    ]

    if use_blas:
        ext_modules += \
        [
            Pybind11Extension(
                "fast_quantum_blas",
                ["undi/fast/permutations.cpp", "undi/fast/fast_blas.cpp"],
                extra_compile_args='-O3 -fopenmp -ffast-math -g'.split(),
                extra_link_args='-O3 -fopenmp -ffast-math -g'.split() + MKL_L.split(),
                include_dirs=[MKL_I]
            ),
        ]
    

except ImportError:
    print("No Pybind11 => This is no good!")
    ext_modules = []
    build_ext = None


setup(name='undi',
      version='1.1',
      description='Muon-Nuclear Dipolar Interaction',
      long_description=open('README.rst').read(),
      classifiers=['License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
                   'Programming Language :: Python :: 3 :: Only',
                   'Operating System :: OS Independent',
                   'Intended Audience :: Science/Research',
                   'Topic :: Scientific/Engineering :: Physics'],
      keywords='muSR spin nuclear',
      url='http://github.com/bonfus/undi',
      author='Pietro Bonfa',
      author_email='bonfus@gmail.com',
      license='GPLv3',
      packages=['undi', 'undi.fast'],
      include_package_data=True,
      install_requires = [
                          'numpy',
                         ],
      extras_require = {
        'Fast C++ implementation':  ["pybind11"],
        'Progress bar':  ["tqdm"]
      },
      zip_safe=False,
      ext_modules=ext_modules,
      headers=glob('undi/fast/*.hpp') if ext_modules else None,
      cmdclass={"build_ext": build_ext}
    )
