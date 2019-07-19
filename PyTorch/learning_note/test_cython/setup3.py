from distutils.core import setup
from Cython.Build import cythonize
setup(
    name='compute_module',
    ext_modules=cythonize('compute3.pyx'),
)

# python setup3.py build