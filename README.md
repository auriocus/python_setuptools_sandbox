Setuptools Sandbox
===================

This project explores the best way to distribute a mixed Python / C package where the C code is parallelized with OpenMP and uses numpy. It should run at least on the major 3 platforms (Windows, macOS, Linux) and uses Github Actions to build for multiple Python versions (via (cibuildwheel)[https://github.com/pypa/cibuildwheel].

Rationale
=========

Python packaging is a ruddy mess, especially when it comes to C extensions and non-standard compiler flags, such as OpenMP. There are no ready made macros for setuptools, not even simple ways to test for compiler flags comparable to autotools or CMake. Rather, one has to dig through arcane stackoverflow hacks to find this information. Therefore, this repo is a place to test out different strategies to get it built withour cluttering the real work.

Soon we will have to start over again when in Python 3.12 distutils will be removed and so far there is no simple solution to run the compiler from setuptools 😠
