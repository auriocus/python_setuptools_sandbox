#! /usr/bin/env python3

import os
import re
import sys
import sysconfig
import platform
import subprocess

from distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages

setup(
    name='setuptools_sandbox',
    version='0.1',
    author='Christian Gollwitzer',
    author_email='auriocus@gmx.de',
    description='A python module with a C extension',
    long_description='',
    packages=find_packages('src'),
    package_dir={'':'src'},
    zip_safe=False,
)
