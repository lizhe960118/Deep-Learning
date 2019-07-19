from distutils.core import setup 
from Cython.Build import cythonize

setup (
    name = "compute_module",
    ext_modules=cythonize('compute1.pyx'),
)