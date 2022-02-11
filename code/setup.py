from setuptools import setup, find_packages

version = 0.1

DISTNAME = 'slope'
LICENSE = 'BSD (3-clause)'
VERSION = version

setup(name='slope',
      install_requires=['numpy>=1.12', 'scikit-learn>=1.0'],
      license=LICENSE,
      packages=find_packages()
      )
