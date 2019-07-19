from distutils.core import setup
from Cython.Build import cythonize
setup(
    name='compute_module',
    ext_modules=cythonize('compute2.pyx'),
)

# python setup2.py build