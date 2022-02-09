#! /usr/bin/env python3

import os
import sys
import sysconfig
import platform
import subprocess
import tempfile
import shutil

from setuptools import Extension, find_packages
from numpy.distutils.core import setup


def ac_check_flag(flags, script):
    # emulate AC_CHECK_FLAG from autotools to test if the compiler
    # supports a given flag. Return the first working flag. 

    # Get compiler invocation
    compiler = os.environ.get('CC',
                              sysconfig.get_config_var('CC'))

    for flag in flags:
        # Create a temporary directory
        tmpdir = tempfile.mkdtemp()
        curdir = os.getcwd()
        os.chdir(tmpdir)

        # Attempt to compile the test script.
        filename = r'flagtest.c'
        with open(filename,'w') as f :
            f.write(script)
        
        try:
            with open(os.devnull, 'w') as fnull:
                compiler_result = subprocess.run([compiler, flag, filename], capture_output=True)
                #print(compiler_result)
                exit_code = compiler_result.returncode
        except OSError :
            exit_code = 1

        # Clean up
        os.chdir(curdir)
        shutil.rmtree(tmpdir)
        
        if exit_code == 0:
            return flag

    return ""


def check_for_openmp():
    """Check  whether the default compiler supports OpenMP.
    This routine is adapted from yt, thanks to Nathan
    Goldbaum. See https://github.com/pynbody/pynbody/issues/124"""
        
    omptestprog = '''
        #include <omp.h>
        #include <stdio.h>
        int main() {
            #pragma omp parallel
            printf("Hello from thread %d, nthreads %d\\n", omp_get_thread_num(), omp_get_num_threads());
        }
'''
    
    ompflags = ['-fopenmp', '/openmp']
    
    ompflag  = ac_check_flag(ompflags, omptestprog);

    if ompflag == "":
        print ("""WARNING
OpenMP support is not available in your default C compiler
The program will only run on a single core. 
""")
        if platform.uname()[0]=='Darwin':
            print ("""Since you are running on Mac OS, it's likely that the problem here
is Apple's Clang, which does not support OpenMP at all. The easiest
way to get around this is to download the latest version of gcc from
here: http://hpc.sourceforge.net. After downloading, just point the
CC environment variable to the real gcc and OpenMP support should
get enabled automatically. Something like this -
sudo tar -xzf /path/to/download.tar.gz /
export CC='/usr/local/bin/gcc'
python setup.py clean
python setup.py build
""")
            print ("""Continuing your build without OpenMP...\n""")

        return []

    return [ompflag]


print("Checking for OpenMP support...\t")

extraompflag = check_for_openmp()

print(extraompflag)

setup(
    name='setuptools_sandbox',
    version='0.1',
    author='Christian Gollwitzer',
    author_email='auriocus@gmx.de',
    description='A python module with a C extension',
    long_description='',
    packages=find_packages('src'),
    package_dir={'':'src'},
    ext_modules=[Extension('setuptools_sandbox/addfloats', ['src/setuptools_sandbox/some_ccode.c'], 
        extra_compile_args = extraompflag,
        extra_link_args = extraompflag)],
    zip_safe=False,
)
