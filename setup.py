from distutils.core import setup
from Cython.Build import cythonize

#ext_modules = cythonize('*.pyx'),
#extra_link_args = ['-stdlib=libc++']
#for e in ext_modules:
#    e.extra_link_args.extend(extra_link_args)

setup(
    name = "rectangleapp",
    ext_modules=cythonize('*.pyx')
)

