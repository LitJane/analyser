import logging
import warnings

from analyser.charter_parser import CharterDocument
from analyser.contract_parser import ContractDocument
from analyser.legal_docs import LegalDocument, Paragraph, PARAGRAPH_DELIMITER, GenericDocument
from analyser.ml_tools import SemanticTag
from analyser.protocol_parser import ProtocolDocument


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
    doc = GenericDocument('')

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
