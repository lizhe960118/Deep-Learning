from distutils.core import setup
from Cython.Build import cythonize
setup(
    name='Hello pyx',
    ext_modules=cythonize('hello_code.pyx')
)
