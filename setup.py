#! /usr/bin/env python3
import os
import sys
import sysconfig
import platform
import subprocess
import tempfile
import shutil

import numpy
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext


def ac_check_flag(flags, script):
    # emulate AC_CHECK_FLAG from autotools to test if the compiler
    # supports a given flag. Return the first working flag. 

    # Get compiler invocation
    compiler = os.environ.get('CC',
                              sysconfig.get_config_var('CC')).split()

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
                compiler_result = subprocess.run(compiler + [flag, filename], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
        return []

    return [ompflag]



class build_ext(_build_ext):
    # find openMP options, if available
    def finalize_options(self):
        _build_ext.finalize_options(self)
        print("Checking for OpenMP support...\t", end="")

        extraompflag = check_for_openmp()

        print(" ".join(extraompflag))

        if extraompflag == []:
            print ("""WARNING
        OpenMP support is not available in your default C compiler
        The program will only run on a single core. 
        """)
        
        for ext in self.extensions:
            ext.extra_compile_args.extend(extraompflag)
            ext.extra_link_args.extend(extraompflag)
            print("Current value: ",ext.extra_compile_args)
            pass


numpyinclude = numpy.get_include()

setup(
    name='setuptools_sandbox',
    version='0.1',
    author='Christian Gollwitzer',
    author_email='auriocus@gmx.de',
    description='A python module with a C extension',
    long_description='',
    packages=find_packages('src'),
    package_dir={'':'src'},
    ext_modules=[Extension('setuptools_sandbox.addfloats', ['src/setuptools_sandbox/some_ccode.c'], 
        include_dirs=[numpyinclude])],
    
    cmdclass={'build_ext':build_ext},
    zip_safe=False,
)
