from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [Extension("hello_code",["hello_code.pyx"])]
setup(
    name = "Hello pyx",
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
