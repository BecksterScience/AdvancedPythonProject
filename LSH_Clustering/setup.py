# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    name="fastlsh",
    ext_modules=cythonize([
        Extension(
            name="fastlsh",
            sources=["fastlsh.pyx"],
            include_dirs=[numpy.get_include()],
        )
    ]),
)