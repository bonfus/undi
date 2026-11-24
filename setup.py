from setuptools import setup
import os
import shlex
import subprocess
import sys
import sysconfig
from glob import glob

use_blas = False

MKL_L = os.getenv("MKL_LIB", "-L/opt/intel/oneapi/mkl/latest/lib/intel64")
# For GNU:
MKL_L += " -lmkl_gf_lp64  -lmkl_gnu_thread -lmkl_core -D__MKL"
# For Intel:
# MKL_L += " -lmkl_intel_lp64  -lmkl_intel_thread -lmkl_core -D__MKL"
MKL_I = os.getenv("MKL_INCLUDE", "/opt/intel/oneapi/mkl/latest/include/")


def _compiler_supports_openmp():
    """True if the C++ compiler accepts -fopenmp and finds omp.h."""
    cxx = shlex.split(os.getenv("CXX") or sysconfig.get_config_var("CXX") or "c++")
    try:
        result = subprocess.run(
            cxx + ["-fopenmp", "-x", "c++", "-fsyntax-only", "-"],
            input="#include <omp.h>\nint main() { return 0; }\n",
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _find_libomp():
    """Find libomp (needed on macOS, where Apple clang has no built-in OpenMP)."""
    candidates = [os.getenv("LIBOMP_PREFIX"), os.getenv("OpenMP_ROOT"), os.getenv("CONDA_PREFIX")]
    try:
        brew = subprocess.run(["brew", "--prefix", "libomp"],
                              capture_output=True, text=True, timeout=60)
        if brew.returncode == 0:
            candidates.append(brew.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    candidates += ["/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"]

    for prefix in candidates:
        if prefix and os.path.isfile(os.path.join(prefix, "include", "omp.h")) \
                and glob(os.path.join(prefix, "lib", "libomp.*")):
            return prefix
    return None


def openmp_flags():
    """Return (compile_args, link_args) to enable OpenMP with the current compiler.

    The compiler's own -fopenmp is used when available. Otherwise, on macOS,
    libomp is used (install it with `brew install libomp`, or set LIBOMP_PREFIX).
    """
    if _compiler_supports_openmp() or sys.platform != "darwin":
        return ["-fopenmp"], ["-fopenmp"]

    prefix = _find_libomp()
    if prefix is None:
        raise RuntimeError(
            "The C++ compiler has no OpenMP support and libomp was not found. "
            "On macOS install it with `brew install libomp`, "
            "or set LIBOMP_PREFIX to the libomp installation directory."
        )
    lib = os.path.join(prefix, "lib")
    return (["-Xpreprocessor", "-fopenmp", "-I" + os.path.join(prefix, "include")],
            ["-L" + lib, "-lomp", "-Wl,-rpath," + lib])

# Pybind11
try:
    from pybind11.setup_helpers import build_ext, Pybind11Extension
    omp_compile, omp_link = openmp_flags()
    ext_modules = [
        Pybind11Extension(
            "fast_quantum",
            ["undi/fast/permutations.cpp", "undi/fast/fast.cpp"],
            extra_compile_args='-O3 -ffast-math -g'.split() + omp_compile,
            extra_link_args='-O3 -ffast-math -g'.split() + omp_link,
        ),
        Pybind11Extension(
            "fast_quantum_light",
            ["undi/fast/permutations.cpp", "undi/fast/fast_light.cpp"],
            extra_compile_args='-O3 -ffast-math -g'.split() + omp_compile,
            extra_link_args='-O3 -ffast-math -g'.split() + omp_link,
        ),
    ]

    if use_blas:
        ext_modules += \
        [
            Pybind11Extension(
                "fast_quantum_blas",
                ["undi/fast/permutations.cpp", "undi/fast/fast_blas.cpp"],
                extra_compile_args='-O3 -ffast-math -g'.split() + omp_compile,
                extra_link_args='-O3 -ffast-math -g'.split() + omp_link + MKL_L.split(),
                include_dirs=[MKL_I]
            ),
        ]

    ext_kwargs = {'ext_modules':ext_modules,
                  'headers': glob('undi/fast/*.hpp'),
                  'cmdclass' : {"build_ext": build_ext}
                 }
   

except ImportError:
    print("No Pybind11 => This is no good! Celio's algorithm will be very slow!")
    ext_kwargs = {}


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
      packages=['undi', 'undi.fast', 'undi.kubo_toyabe'],
      include_package_data=True,
      requires = [
                  'setuptools',
                  'numpy',
                  'scipy'
                 ],
      extras_require = {
        'Fast C++ implementation':  ["pybind11"],
        'Progress bar':  ["tqdm"]
      },
      zip_safe=False,
      **ext_kwargs
    )
