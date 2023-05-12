#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


import unittest

from analyser.charter_parser import CharterDocument, CharterParser
from analyser.log import logger
from analyser.parsing import AuditContext
from analyser.runner import get_audits, get_docs_by_audit_id, CHARTER
from integration.db import get_mongodb_connection
from integration.word_document_parser import join_paragraphs


class TestCharterAnalyse(unittest.TestCase):

  @unittest.skipIf(get_mongodb_connection() is None, "requires mongo")
  def test_get_org_names(self):
    parser = CharterParser()

    audits = get_audits()
    if len(audits) == 0:
      logger.warning('no audits')
      return

    audit_id = audits[0]['_id']
    docs = get_docs_by_audit_id(audit_id, kind=CHARTER)

    for db_document in docs:
      print(db_document['filename'])

      parsed_p_json = db_document['parse']
      charter: CharterDocument = join_paragraphs(parsed_p_json, doc_id=db_document['_id'])

      # TODO: mind, this could be slow if embedding is required
      parser.find_org_date_number(charter, AuditContext())




unittest.main(argv=['-e utf-8'], verbosity=3, exit=False)
