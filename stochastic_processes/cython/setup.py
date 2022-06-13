from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(
        "diffusion_processes.pyx",
        compiler_directives={'language_level' : "3"}  # or "2" or "3str"
        ),
)
setup(
    ext_modules = cythonize(
        "market_arrival_processes.pyx",
        compiler_directives={'language_level' : "3"}  # or "2" or "3str"
        ),
)