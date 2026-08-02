from setuptools import setup, Extension
import numpy as np

rng_extension = Extension(
    name='pygifi_rng',
    sources=['pygifi_rng.c'],
    include_dirs=[
        '.',                    # local copies of R headers
        np.get_include(),       # numpy headers
    ],
    define_macros=[
        ('MATHLIB_STANDALONE', '1'),
    ],
    extra_compile_args=['-O2', '-std=c99'],
)

setup(
    name='pygifi_rng',
    version='1.0.0',
    ext_modules=[rng_extension],
)
