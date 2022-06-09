import os
import sys

from distutils.core import setup, Extension
from distutils import sysconfig

cpp_args = ['-std=c++14'] # , '-stdlib=libc++', '-mmacosx-version-min=10.7']

ext_modules = [
    Extension(
        'des_cpp',
        ['des.cpp', 'wrap.cpp'],
        include_dirs=['pybind11/include'],
        language='c++',
        extra_compile_args=cpp_args,
    ),
]

setup(
    name='des_cpp',
    version='0.0.1',
    author='Yufeng Zheng',
    author_email='yufeng_zheng@berkeley.edu',
    description='Multi server queue implemented in C++',
    ext_modules=ext_modules,
)
