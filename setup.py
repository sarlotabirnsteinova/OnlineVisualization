import setuptools
import numpy as np

from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension(
        "dffc.totalvar_cython",
        [
            "src/dffc/ctotalvar.pyx",
            "src/dffc/totalvar.c"
        ],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        #extra_compile_args=["-fno-wrapv",  "-fno-strict-aliasing"],
    ),
]

setup(
    packages=setuptools.find_packages('src'),
    package_dir={"": "src"},
    ext_modules = cythonize(extensions),
)
