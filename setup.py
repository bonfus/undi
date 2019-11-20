from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='undi',
      version='0.1',
      description='Muon-Nuclear Dipolar Interaction',
      classifiers=['License :: OSI Approved :: GPLv3 License',
                   'Programming Language :: Python',
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


