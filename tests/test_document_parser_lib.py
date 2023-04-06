#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
import unittest

from integration.db import get_mongodb_connection
from integration.word_document_parser import WordDocParser
from analyser.legal_docs import LegalDocument, Paragraph
from analyser.ml_tools import SemanticTag
from analyser.text_normalize import normalize_text, replacements_regex

@unittest.skip
class TestContractParser(unittest.TestCase):

  def n(self, txt):
    return normalize_text(txt, replacements_regex)

  @unittest.skipIf(get_mongodb_connection() is None, "requires mongo")
  def test_doc_parser(self):
    get_mongodb_connection()

    FILENAME = "/Users/artem/work/nemo/goil/IN/Другие договоры/Договор Формула.docx"

    wp = WordDocParser()
    res = wp.read_doc(FILENAME)

    doc: LegalDocument = LegalDocument('')
    doc.parse()

    last = 0
    for d in res['documents']:
      for p in d['paragraphs']:
        header_text = p['paragraphHeader']['text'] + '\n'
        body_text = p['paragraphBody']['text'] + '\n'

        header = LegalDocument(header_text)
        header.parse()
        # self.assertEqual(self.n(header_text), header.text)

        doc += header
        headerspan = (last, len(doc.tokens_map))
        print(headerspan)
        last = len(doc.tokens_map)

        body = LegalDocument(body_text)
        body.parse()
        doc += body
        bodyspan = (last, len(doc.tokens_map))

        header_tag = SemanticTag('headline', header_text, headerspan)
        body_tag = SemanticTag('paragraphBody', None, bodyspan)

        print(header_tag)
        # print(body_tag)
        para = Paragraph(header_tag, body_tag)
        doc.paragraphs.append(para)
        last = len(doc.tokens_map)

        _ = doc.subdoc_slice(para.header.as_slice())
        _ = doc.subdoc_slice(para.body.as_slice())
        # self.assertEqual(self.n(header_text), h_subdoc.text)
        # self.assertEqual(self.n(body_text), b_subdoc.text)

    print('-' * 100)
    print(doc.text)

    _ = [doc.subdoc_slice(p.header.as_slice()) for p in doc.paragraphs]
    print('-' * 100)


unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
