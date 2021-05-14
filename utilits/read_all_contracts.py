#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
import json
import sys
import traceback

import pymongo
from pandas import DataFrame

from analyser.documents import TOKENIZER_DEFAULT
from analyser.legal_docs import LegalDocument
from integration.word_document_parser import WordDocParser, join_paragraphs


def read_all_docs(files_dir: str, doc_type='CONTRACT'):
  db = get_mongodb_connection()
  collection = db['legaldocs']

  wp = WordDocParser()
  filenames = wp.list_filenames(files_dir)

  cnt = 0
  failures = 0
  unknowns = 0
  nodate = 0

  def stats():
    print(f'processed:{cnt};\t failures:\t{failures}\t unknown type: {unknowns}\t unknown date: {nodate}')

  for fn in filenames:

    shortfn = fn.split('/')[-1]
    pth = '/'.join(fn.split('/')[5:-1])
    _doc_id = pth + '/' + shortfn

    cnt += 1
    print(cnt, fn)

    docs = collection.find_one({"_id": _doc_id, 'version': wp.version})
    if docs is None:
      # parse and save to DB
      try:
        docs = wp.read_doc(fn)

        # for res in docs['documents']:
        docs['short_filename'] = shortfn
        docs['path'] = pth
        docs['_id'] = _doc_id

        collection.delete_many({"_id": _doc_id})
        collection.insert_one(docs)

      except Exception:
        print(f"{fn}\nException in WordDocParser code:")
        traceback.print_exc(file=sys.stdout)
        failures += 1

    if docs:
      for res in docs['documents']:

        if doc_type == res["documentType"]:
          _doc = _parse_doc(res, _doc_id)
          yield _doc

          # if False:
          #   contract: ContractDocument = _parse_contract(_doc, row)
          #   json_struct = DocumentJson(contract).__dict__
          #
          #   if contracts_collection.find_one({"_id": _doc_id}) is None:
          #     contracts_collection.insert_one(json_struct)

  stats()
  # print(f'failures:\t{failures}\t unknowns: {unknowns}\t nodate: {nodate}' )


def _parse_doc(res, doc_id) -> LegalDocument:
  return join_paragraphs(res, doc_id)


def dump_contracts_from_db_to_jsons(output_path):
  db = get_mongodb_connection()
  collection = db['legaldocs']

  wp = WordDocParser()
  filenames = wp.list_filenames('/Users/artem/Downloads/Telegram Desktop/X0/')

  for fn in filenames:
    print(fn)
    shortfn = fn.split('/')[-1]
    pth = '/'.join(fn.split('/')[5:-1])
    _doc_id = pth + '/' + shortfn

    res = collection.find_one({"_id": _doc_id})
    if res is not None:
      json_name = _doc_id.replace('/', '_')
      with open(f'{output_path}/{json_name}.json', 'w') as file:
        _j = json.dumps(res, indent=4, ensure_ascii=False, default=lambda o: '<not serializable>')
        file.write(_j)
        print(f'saved file to {json_name}')


from integration.db import get_mongodb_connection
from analyser.doc_structure import get_tokenized_line_number


def analyse_headers():
  db = get_mongodb_connection()
  collection = db['legaldocs']
  headers_collection = db['headers']
  headers_collection.drop()
  headers_collection = db['headers']

  res = collection.find({})
  k = 0
  for doc in res:
    print(k)
    k += 1

    for p in doc['paragraphs']:
      header_text = p['paragraphHeader']['text']

      tokens = TOKENIZER_DEFAULT.tokenize(header_text)
      _, span, _, _ = get_tokenized_line_number(tokens, 0)
      header_id = ' '.join(tokens[span[1]:])
      header_id = header_id.lower()

      # header_id = header_text

      existing = headers_collection.find_one({'text': header_id})
      if existing:
        existing['count'] += 1
        headers_collection.update({'_id': existing['_id']}, existing, True)
        pass
      else:
        header = {
          # '_id': header_id,
          'doc_id': doc['_id'],
          'text': header_id,
          'len': len(header_text),
          'has_newlines': header_text.count('\n'),
          'count': 1
        }
        headers_collection.insert_one(header)
      # headers_collection.update_one({'_id': header_id}, header, True)

  """
  ACHTUNG! ["] not found with text.find, next text is: ``
55о 05`00``
в.д.
Точка №11

  """


def find_top_headers():
  db = get_mongodb_connection()
  headers_collection = db['headers']
  q = {
    'count': {
      '$gt': 10
    }
  }
  items = []
  for c in headers_collection.find(q).sort('count', pymongo.DESCENDING):
    items.append({
      'text': c['text'],
      'count': c['count'],
      'doc_id': c['doc_id']
    })
    print(c['count'], '\t', c['text'])
  df = DataFrame.from_records(items)
  df.to_csv('top_headers.csv')


if __name__ == '__main__':
  # find_top_headers()
  # analyse_headers()
  read_all_docs('_files_dir')
  # dump_contracts_from_db_to_jsons('/Users/artem/work/nemo/goil/OUT/jsons')
