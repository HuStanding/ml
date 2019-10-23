# -*- coding: utf-8 -*-
# @Author: huzhu
# @Date:   2019-10-23 17:14:10
# @Last Modified by:   huzhu
# @Last Modified time: 2019-10-23 19:33:41

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
setup(ext_modules = cythonize(Extension(
    'kd_tree',
    sources=['kd_tree.pyx'],
    language='c',
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[]
)))

