#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from charter_parser import CharterDocumentParser
from charter_patterns import CharterPatternFactory
from contract_parser import ContractAnlysingContext
from legal_docs import *
from parsing import print_prof_data
from patterns import *


class FakeEmbedder(AbstractEmbedder):

  def __init__(self, default_point):
    self.default_point = default_point

  def embedd_tokenized_text(self, tokenized_sentences_list, lens):
    # def get_embedding_tensor(self, tokenized_sentences_list):
    tensor = []
    for sent in tokenized_sentences_list:
      sentense_emb = []
      for token in sent:
        token_emb = self.default_point
        sentense_emb.append(token_emb)
      tensor.append(sentense_emb)

    return np.array(tensor), tokenized_sentences_list


class LegalDocumentTestCase(unittest.TestCase):

  def test_embedd_large(self):
    point1 = [1, 6, 4]
    emb = FakeEmbedder(point1)

    ld = LegalDocument('a b c d e f g h')

    ld.parse()
    print(ld.tokens)
    ld._embedd_large(emb, 5)

    # print(ld.embeddings)
    print(ld.tokens)


  def test_parse(self):
    d = LegalDocument("a")
    d.parse()
    print(d.tokens)
    self.assertEqual(1, len(d.tokens))

  def test_analyze_contract_0(self):

    point1 = [1, 6, 4]
    emb = FakeEmbedder(point1)

    ctx = ContractAnlysingContext(emb )
    ctx.analyze_contract("1. ЮРИДИЧЕСКИЙ содержание 4.")

    ctx._logstep("analyze_contract")

  def test_charter_parser(self):
    # from renderer import SilentRenderer
    point1 = [1, 6, 4]

    cpf = CharterPatternFactory(FakeEmbedder(point1))
    ctx = CharterDocumentParser(cpf)

    ctx.analyze_charter("1. ЮРИДИЧЕСКИЙ содержание 4.")
    ctx._logstep("analyze_charter")


if __name__ == '__main__':
  unittest.main()
