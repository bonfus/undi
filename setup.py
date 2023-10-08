from setuptools import setup
import os

# Pybind11
try:
    from pybind11.setup_helpers import build_ext, Pybind11Extension
    ext_modules = [
        Pybind11Extension(
            "fast_quantum",
            ["undi/fast.cpp"],
            extra_compile_args='-fopenmp'.split(),
            extra_link_args='-fopenmp'.split(),
        ),
    ]

except ImportError:
    print("No Pybind11 => This is no good!")
    ext_modules = []
    build_ext = None

# MPI
if not (build_ext is None):
    try:
        import mpi4py as m;
        print("Using MPI: " + m.__version__); print("MPI include: " + m.get_include());

        ext_modules += [
            Pybind11Extension(
                "fast_quantum_mpi",
                ["undi/fast_mpi.cpp"],
                extra_compile_args='-fopenmp'.split(),
                extra_link_args='-fopenmp'.split(),
                include_dirs=[m.get_include()]
            ),
        ]
    except ImportError:
        print("No MPI detected")


def readme():
    with open('README.rst') as f:
        return f.read()

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
      packages=['undi'],
      include_package_data=True,
      install_requires = [
                          'numpy',
                          'qutip',
                          'mendeleev'
                          ],
      extras_require = {
        'Fast C++ implementation':  ["pybind11"],
        'Progress bar':  ["tqdm"]
      },
      zip_safe=False,
      ext_modules=ext_modules,
      cmdclass={"build_ext": build_ext}
    )
