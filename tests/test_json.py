#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import os
import pickle
import unittest

from analyser.contract_parser import ContractParser, ContractDocument
from analyser.contract_patterns import ContractPatternFactory
from analyser.documents import TextMap
from analyser.legal_docs import DocumentJson
from analyser.ml_tools import SemanticTag
from analyser.parsing import AuditContext
from analyser.schemas import ContractSchema


class TestJsonExport(unittest.TestCase):

  def _get_doc(self) -> (ContractDocument, ContractPatternFactory):
    pth = os.path.dirname(__file__)
    with open(pth + '/2. Договор по благ-ти Радуга.docx.pickle', 'rb') as handle:
      doc = pickle.load(handle)

    with open(pth + '/contract_pattern_factory.pickle', 'rb') as handle:
      factory = pickle.load(handle)

    self.assertEqual(1024, doc.embeddings.shape[-1])

    return doc, factory

  def _get_doc_factory_ctx(self):
    doc, factory = self._get_doc()

    ctx = ContractParser(embedder={})
    ctx.verbosity_level = 3

    return doc, factory, ctx

  def print_semantic_tag(self, tag: SemanticTag, map: TextMap):
    print('print_semantic_tag:', tag, f"[{map.text_range(tag.span)}]")

  def test_to_json(self):
    doc, _, ctx = self._get_doc_factory_ctx()

    doc.__dict__['number'] = None  # hack for old pickles
    doc.__dict__['date'] = None  # hack for old pickles
    doc.__dict__['warnings'] = []  # hack for old pickles
    doc.__dict__['attributes_tree'] = ContractSchema()  # hack for old pickles

    actx = AuditContext()
    ctx.find_attributes(doc, actx)
    json_struct = DocumentJson(doc)
    _j = json_struct.dumps()

    # TODO: compare with file


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
