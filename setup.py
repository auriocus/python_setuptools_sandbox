#! /usr/bin/env python3

import sys
import sysconfig
import platform
import subprocess

from setuptools import Extension, find_packages
from numpy.distutils.core import setup

setup(
    name='setuptools_sandbox',
    version='0.1',
    author='Christian Gollwitzer',
    author_email='auriocus@gmx.de',
    description='A python module with a C extension',
    long_description='',
    packages=find_packages('src'),
    package_dir={'':'src'},
    ext_modules=[Extension('setuptools_sandbox/addfloats', ['src/setuptools_sandbox/some_ccode.c'])],
    zip_safe=False,
)
