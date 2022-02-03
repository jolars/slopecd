from setuptools import setup

version = 0.1

DISTNAME = 'slope'
LICENSE = 'BSD (3-clause)'
VERSION = version

setup(name='slope',
      install_requires=['numpy>=1.12'],
      license=LICENSE,
      packages=['slope'])
