from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='undi',
      version='1.0',
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
      zip_safe=False)
