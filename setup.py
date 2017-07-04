from distutils.core import setup
from distutils.extension import Extension
from sys import platform

import numpy
from Cython.Distutils import build_ext

exec(open('version.py').read())

set_dist_ext = Extension("uai2017experiments.cython_impl.cy_set_dist",
                         sources=['uai2017experiments/cython_impl/cy_set_dist.pyx', 'uai2017experiments/cython_impl/c_cy_set_dist.cpp'],
                         language='c++',
                         extra_compile_args=["-std=c++11", "-stdlib=libc++",
                                             "-mmacosx-version-min=10.7"] if platform == "darwin" else ["-std=c++11"],
                         extra_link_args=["-std=c++11", "-stdlib=libc++",
                                          "-mmacosx-version-min=10.7"] if platform == "darwin" else ["-std=c++11"],
                         include_dirs=[numpy.get_include()])

setup(
    name='KRCIT',
    packages=['uai2017experiments'],
    version=__version__,
    author='Sanghack Lee',
    author_email='sanghack.lee@gmail.com',

    cmdclass={'build_ext': build_ext},
    ext_modules=[set_dist_ext],

)
#  python3 setup.py build_ext --inplace
#  pip install -e .
