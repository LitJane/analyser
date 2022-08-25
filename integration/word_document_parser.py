import json
import logging
import os
import subprocess
import warnings

from analyser.charter_parser import CharterDocument
from analyser.contract_parser import ContractDocument
from analyser.legal_docs import LegalDocument, Paragraph, PARAGRAPH_DELIMITER
from analyser.log import logger
from analyser.ml_tools import SemanticTag
from analyser.protocol_parser import ProtocolDocument
from integration.doc_providers import DirDocProvider


class WordDocParser(DirDocProvider):
  version = '1.1.19'

  def __init__(self):

    x = os.system("java -version")

    if x != 0:
      raise RuntimeError(f'java executable returned {x}')

    if 'documentparser' in os.environ:
      self.documentparser = os.environ['documentparser']
    else:
      msg = f'please set "documentparser" environment variable to point ' \
            f'document-parser-{self.version} unpacked lib ' \
            f'(downloadable from https://github.com/nemoware/document-parser)'
      warnings.warn(msg)

      self.documentparser = f'../libs/document-parser-{self.version}'

    self.cp = f"{self.documentparser}/classes:{self.documentparser}/lib/*"
    logger.info(self.cp)

  def read_doc(self, fn) -> dict:
    fn = fn.encode('utf-8')

    if not os.path.isfile(fn):
      raise ValueError(f'{fn} does not exist')

    s = ["java", "-cp", self.cp, "com.nemo.document.parser.App", "-i", fn]
    result = subprocess.run(s, stdout=subprocess.PIPE, encoding='utf-8')

    if result.returncode != 0:
      raise RuntimeError(f'cannot execute {result.args}')

    return json.loads(result.stdout)


def create_doc_by_type(t: str, doc_id, filename) -> CharterDocument or ContractDocument or ProtocolDocument:
  # TODO: check type of res

  if t in ('CONTRACT', 'ANNEX', 'SUPPLEMENTARY_AGREEMENT', 'AGREEMENT'):
    doc = ContractDocument('')
  elif t == 'PROTOCOL':
    doc = ProtocolDocument()
  elif t == 'CHARTER':
    doc = CharterDocument()
  else:
    logging.warning(f"Unsupported document type: {t}")
    doc = LegalDocument('')

  doc._id = doc_id
  doc.filename = filename

  doc.parse()
  return doc


def join_paragraphs(response, doc_id, filename=None) -> CharterDocument or ContractDocument or ProtocolDocument:
  # TODO: check type of res

  doc = create_doc_by_type(response['documentType'], doc_id, filename)

  fields = ['documentType']
  for key in fields:
    doc.__setattr__(key, response.get(key, None))

  last = 0
  # remove empty headers
  paragraphs = []
  for _p in response['paragraphs']:
    header_text = _p['paragraphHeader']['text']
    if header_text.strip() != '':
      paragraphs.append(_p)
    else:
      doc.warnings.append('blank header encountered')
      warnings.warn('blank header encountered')

  for _p in paragraphs:
    header_text = _p['paragraphHeader']['text']
    header_text = header_text.replace('\n', ' ').strip() + PARAGRAPH_DELIMITER

    header = LegalDocument(header_text)
    header.parse()

    doc += header
    headerspan = (last, len(doc.tokens_map))

    last = len(doc.tokens_map)

    if _p['paragraphBody']:
      body_text = _p['paragraphBody']['text'] + PARAGRAPH_DELIMITER
      appendix = LegalDocument(body_text).parse()
      doc += appendix

    bodyspan = (last, len(doc.tokens_map))

    header_tag = SemanticTag('headline', header_text, headerspan)
    body_tag = SemanticTag('paragraphBody', None, bodyspan)

    para = Paragraph(header_tag, body_tag)
    doc.paragraphs.append(para)
    last = len(doc.tokens_map)

  doc.split_into_sentenses()
  return doc


if __name__ == '__main__':
  wp = WordDocParser()
  res = wp.read_doc("/Users/artem/work/nemo/goil/IN/Другие договоры/Договор Формула.docx")
  for d in res['documents']:
    print("-" * 100)
    for p in d['paragraphs']:
      print(' 📃 ', p['paragraphHeader']['text'])

  print("=" * 100)
  for d in res['documents']:
    print("-" * 100)
    c = join_paragraphs(d, 'Другие договоры/Договор Формула.docx')
    print(c.text)
    print(c.__dict__.keys())
