"""FLow54: A compressible flow library."""

from setuptools import setup
import os
import sys

setup(name = 'Flow54',
    version = '1.0.0',
    description = "FLow54: A compressible flow library.",
    url = 'https://github.com/corygoates/Flow54',
    author = 'Cory Goates',
    author_email = 'cory.goates@usu.edu',
    install_requires = ['numpy>=1.18', 'scipy>=1.4', 'pytest', 'matplotlib'],
    python_requires ='>=3.6.0',
    license = 'MIT',
    packages = ['flow54'],
    zip_safe = False)
