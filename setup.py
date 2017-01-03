from __future__ import print_function
import sys
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

setup(name='convnetskeras',
      version='0.1',
      description='Pre-trained convnets in keras',
      author='Leonard Blier',
      author_email='leonard.blier@ens.fr',
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      )
