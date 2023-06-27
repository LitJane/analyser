#!/usr/bin/python1.
# -*- coding: utf-8 -*-
# coding=utf-8
from analyser.hyperparams import HyperParameters
from analyser.log import logger

__version__ = "23.6.27"  # year.month.day.minor

__version_ints__ = [int(x) for x in __version__.split('.')]
_msg = f'Nemoware Analyser v{__version__}'
print(_msg)
logger.info(_msg)

HyperParameters()
