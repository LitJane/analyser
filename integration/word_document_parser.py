import datetime
import json
import os
import subprocess
import warnings

from contract_parser import ContractDocument
from integration.doc_providers import DirDocProvider
from legal_docs import LegalDocument, Paragraph, PARAGRAPH_DELIMITER
from ml_tools import SemanticTag
from protocol_parser import ProtocolDocument


class WordDocParser(DirDocProvider):

  def __init__(self):

    self.version = '1.1.10'

    x = os.system("java -version")
    assert x == 0
    if 'documentparser' in os.environ:
      self.documentparser = os.environ['documentparser']
    else:
      msg = f'please set "documentparser" environment variable to point ' \
            f'document-parser-{self.version} unpacked lib ' \
            f'(downloadable from https://github.com/nemoware/document-parser)'
      warnings.warn(msg)

      self.documentparser = f'../libs/document-parser-{self.version}'

    self.cp = f"{self.documentparser}/classes:{self.documentparser}/lib/*"
    print(self.cp)

  def read_doc(self, fn) -> dict:
    fn = fn.encode('utf-8')

    assert os.path.isfile(fn), f'{fn} does not exist'

    s = ["java", "-cp", self.cp, "com.nemo.document.parser.App", "-i", fn]
    result = subprocess.run(s, stdout=subprocess.PIPE, encoding='utf-8')

    res = json.loads(result.stdout)

    return res


def join_paragraphs(response, doc_id):
  # TODO: check type of res
  doc = None
  if response['documentType'] == 'CONTRACT':
    doc: LegalDocument = ContractDocument('')
  elif response['documentType'] == 'PROTOCOL':
    doc: LegalDocument = ProtocolDocument(None)
  else:
    warnings.warn("Unsupported document type:", response['documentType'])
    doc: LegalDocument = LegalDocument('')

  doc.parse()

  fields = ['documentDate', 'documentNumber', 'documentType']

  for key in fields:
    doc.__dict__[key] = response[key]

  last = 0

  for p in response['paragraphs']:

    header_text = p['paragraphHeader']['text'] + PARAGRAPH_DELIMITER
    header_text = header_text.replace('\n', ' ')

    header = LegalDocument(header_text)
    header.parse()

    doc += header
    headerspan = (last, len(doc.tokens_map))

    last = len(doc.tokens_map)

    if p['paragraphBody']:
      body_text = p['paragraphBody']['text'] + PARAGRAPH_DELIMITER
      body = LegalDocument(body_text)
      body.parse()
      doc += body

    bodyspan = (last, len(doc.tokens_map))

    header_tag = SemanticTag('headline', header_text, headerspan)
    body_tag = SemanticTag('paragraphBody', None, bodyspan)

    para = Paragraph(header_tag, body_tag)
    doc.paragraphs.append(para)
    last = len(doc.tokens_map)

  # doc.parse()

  if response["documentDate"]:
    doc.date = datetime.datetime.strptime(response["documentDate"], '%Y-%m-%d')

  doc._id = doc_id
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
