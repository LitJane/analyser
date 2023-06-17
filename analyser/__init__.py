#!/usr/bin/python1.
# -*- coding: utf-8 -*-
# coding=utf-8
from analyser.hyperparams import HyperParameters

__version__ = "23.6.13"  # year.month.day.minor

__version_ints__ = [int(x) for x in __version__.split('.')]
print(f'Nemoware Analyser v{__version__}')

HyperParameters()
