import numpy as np
from setuptools import Extension, setup


extensions = [
    Extension(
        "dynflatfield.totalvar_cython",
        [
            "src/dynflatfield/ctotalvar.pyx",
            "src/dynflatfield/totalvar.c"
        ],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

setup(
    ext_modules=extensions,
)
