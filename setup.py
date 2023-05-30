#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
from pip._internal.req import parse_requirements
from setuptools import setup, find_namespace_packages
from __init__ import __version__

import analyser

with open("README.md", "r") as fh:
  long_description = fh.read()


# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('requirements.txt', session='hack')

# reqs is a list of requirement
reqs = []
for ir in install_reqs:
  if 'rec' in ir.__dict__:
    reqs.append(str(ir.req))
  else:
    reqs.append(str(ir.requirement))


setup(
  name="nemoware-analyzer",
  version=__version__,

  description="GPN Audit: NLP analyser",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/nemoware/analyser",


  install_requires=reqs,
  packages=find_namespace_packages(exclude=["tests", "notebooks"]),
  include_package_data=True,
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  python_requires='>=3.6',



  entry_points={"console_scripts": ["analyser-run = bin.analyser_run:main"]},
  scripts=["bin/analyser-run.cmd"],
)