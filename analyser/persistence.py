import datetime
import warnings

import numpy as np

from analyser.documents import TextMap, CaseNormalizer
from analyser.legal_docs import Paragraph
from analyser.ml_tools import SemanticTag
from analyser.structures import DocumentState
from integration.word_document_parser import join_paragraphs, create_doc_by_type


class DbJsonDoc:

  def __init__(self, j: dict):
    self.analysis = None
    self.user = None

    self.parserResponseCode: int = -1
    self.state: int = -1
    self.parse = None
    self.filename = None

    self._id = None
    self.isActive: bool or None = None
    self.retry_number: int = 0
    self.__dict__.update(j)  # ------important
    self.documentType = self.parse['documentType']

    if self.state == DocumentState.New.value:
      # reset user data, because it is bound to tokenisation map, re-tokenisation is possible
      self.user = None
      self.analysis = None

  def get_id(self):
    return self._id

  def isPreprocessed(self):
    return self.state == DocumentState.Preprocessed.value or self.state == DocumentState.Error.value

  def isNew(self):
    return self.state == DocumentState.New.value or self.state is None or self.state == DocumentState.Unknown.value

  def isActiveCharter(self):
    return (self.isActive or self.isActive is None) and self.documentType == 'CHARTER'

  def is_user_corrected(self) -> bool:
    if self.state == DocumentState.New.value:
      return False
    return self.user is not None and self.user.get('attributes', None) is not None

  def asLegalDoc(self):

    if self.is_analyzed():
      # attributes are bound to an existing tokens map
      # -->  preserve saved tokenization
      doc = create_doc_by_type(self.parse['documentType'], self._id, filename=self.filename)

      doc.tokens_map_norm = self.get_tokens_for_embedding()
      doc.tokens_map = self.get_tokens_map_unchaged()
      if 'sentence_map' in doc.__dict__:
        doc.sentence_map = self.get_sentence_map()
        if doc.sentence_map is None:
          doc.split_into_sentenses()

      headers = self.analysis.get('headers', None)
      if headers is not None:
        doc.paragraphs = []
        last = len(doc.tokens_map)
        for i, h in enumerate(headers):
          header_tag = SemanticTag('headline', h['value'], h['span'])
          body_end = last
          if i < len(headers) - 1:
            body_end = headers[i + 1]['span'][0]
          bodyspan = header_tag.span[1] + 1, body_end
          body_tag = SemanticTag('paragraphBody', None, bodyspan)

          para = Paragraph(header_tag, body_tag)
          doc.paragraphs.append(para)
    else:
      # re-combine parser data
      doc = join_paragraphs(self.parse, self._id, filename=self.filename)
      pass

    doc.user = self.user
    return doc

  def get_tokens_map_unchaged(self) -> TextMap:
    _map = self.analysis['tokenization_maps']['words']
    tokens_map = TextMap(self.analysis['normal_text'], _map)
    return tokens_map

  def get_sentence_map(self) -> TextMap:
    if 'sentences' in self.analysis['tokenization_maps']:
      _map = self.analysis['tokenization_maps']['sentences']
      tokens_map = TextMap(self.analysis['normal_text'], _map)
      return tokens_map

  def get_tokens_for_embedding(self) -> TextMap:
    _tokens_map = self.get_tokens_map_unchaged()
    tokens_map_norm = CaseNormalizer().normalize_tokens_map_case(_tokens_map)
    return tokens_map_norm

  def as_dict(self):
    return self.__dict__

  def __len__(self):
    arrr = self.analysis['tokenization_maps']['words']
    return len(arrr)

  def is_analyzed(self) -> bool:
    # TODO: this is waaaaay to complex
    if self.state == DocumentState.New.value:
      return False

    return ((self.analysis is not None) and (
            self.analysis.get('attributes', None) is not None)) or self.is_user_corrected()

  def get_attributes(self) -> dict:
    warnings.warn("switch to attributes_tree", DeprecationWarning)
    attributes = {}
    if self.user is not None:
      attributes = self.user.get('attributes', {})
    elif self.analysis is not None:
      attributes = self.analysis.get('attributes', {})
    return attributes

  def get_attributes_tree(self) -> dict:
    a = {}
    if self.user is not None:
      a = self.user.get('attributes_tree', {})
      return next(iter(a.values()))
    elif self.analysis is not None:
      a = self.analysis.get('attributes_tree', {})
      return next(iter(a.values()))
    return a

  def get_subject(self) -> dict:
    return self.get_attribute('subject')

  def get_attribute_value(self, attr: str) -> str or None:
    a = self.get_attributes_tree().get(attr)
    if a is not None:
      return a['value']
    return None

  def get_date_value(self) -> datetime.datetime or None:
    return self.get_attribute_value('date')

  def get_attribute_OLD(self, attr) -> dict:
    atts = self.get_attributes()
    if attr in atts:
      return atts[attr]
    else:
      return {
        'value': None,
        'confidence': 0.0,
        'span': [np.nan, np.nan]
      }  ## fallback for safety

  def get_attr_span_start(self, a):
    att = self.get_attributes()
    if a in att:
      return att[a]['span'][0]
