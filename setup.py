#!/usr/bin/env python
from setuptools import setup, find_packages

from os import path
root=path.abspath(path.dirname(__file__))
with open(path.join(root, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
with open(path.join(root,'requirements.txt')) as f:
    requirements = [l.rstrip('\n') for l in f]  

setup(
    author='Forrest Koch',
    author_email='forrest.koch@unsw.edu.au',
    description='Package to compute DCDF for nifti formatted MRI data.',
    entry_points={'console_scripts':['dcdf=dcdf.cli:main']},
    install_requires=requirements,
    keywords='MRI nifti dti neuroimaging dcdf',
    long_description=long_description,
    long_description_content_type='text/markdown',
    maintainer='Forrest Koch',
    maintainer_email='forrest.koch@unsw.edu.au',
    name='dcdf',
    packages=find_packages(),
    url='https://github.com/ForrestCKoch/DCDF',
    version='0.1.1'
)
