#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from charter_parser import CharterDocumentParser
from charter_patterns import CharterPatternFactory
from tf_support.embedder_elmo import ElmoEmbedder


class TestAnalyse(unittest.TestCase):

  # TODO: disabled, because it is slow.
  def ___test_on_smallText(self):
    microsample = """
    Общие положения

    1.1. В соответствии с настоящим Договором Жертвователь обязуется безвозмездно передать Получателю денежные средства в размере 30 000 (Тридцать тысяч) рублей в качестве пожертвования.

    """

    elmo_embedder = ElmoEmbedder()
    CPF = CharterPatternFactory(elmo_embedder)
    CTX = CharterDocumentParser(CPF)
    CTX.analyze_charter(microsample)

    pass


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
