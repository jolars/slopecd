import os
from glob import glob

import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import Extension, setup
import setuptools

__version__ = "0.1.0"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

ext_modules = [
    Pybind11Extension(
        "_slope",
        sources=sorted(glob("slope/src/**/*.cpp", recursive=True)),
        define_macros=[("VERSION_INFO", __version__)],
        cxx_std=14,
    ),
]

setup(
    name="slope",
    version=__version__,
    description="Coordinate Descent Algorithms for SLOPE",
    long_description="",
    ext_modules=ext_modules,
    # packages=setuptools.find_packages(),
    # install_requires=[
    #     "requests",
    #     "importlib-metadata",
    #     "benchopt",
    #     "libsvmdata",
    #     "numba",
    #     "numpy >= 1.12",
    #     "scikit-learn >= 1.0",
    #     "scipy",
    # ],
    license="GPLv3+",
    # extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    # cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
