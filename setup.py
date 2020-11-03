#!/usr/bin/env python
from distutils.core import setup

setup(name='jxf',
      version='0.1',
      description='JAX exponential family support for tensorflow-probability',
      author='Scott Linderman',
      author_email='scott.linderman@stanford.edu',
      url='https://github.com/lindermanlab/jxf',
      install_requires=['jax', 'jaxlib', 'tfp-nightly'],
      packages=['jxf'],
    )
