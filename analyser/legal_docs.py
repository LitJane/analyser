#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


# legal_docs.py
import datetime
import json
import warnings
from enum import Enum

import numpy as np
from bson import json_util
from overrides import final

import analyser
from analyser.attributes import to_json
from analyser.doc_structure import get_tokenized_line_number
from analyser.documents import split_sentences_into_map, TextMap, CaseNormalizer
from analyser.embedding_tools import AbstractEmbedder
from analyser.hyperparams import HyperParameters
from analyser.log import logger
from analyser.ml_tools import SemanticTag, FixedVector, Embeddings, filter_values_by_key_prefix, rectifyed_sum, \
  conditional_p_sum, clean_semantic_tag_copy
from analyser.patterns import DIST_FUNC, AbstractPatternFactory, make_pattern_attention_vector
from analyser.schemas import ContractPrice, DocumentSchema, GenericDocSchema
from analyser.structures import ContractTags
from analyser.text_normalize import normalize_text, replacements_regex
from analyser.text_tools import find_token_before_index
from analyser.transaction_values import _re_greather_then, _re_less_then, _re_greather_then_1, VALUE_SIGN_MIN_TOKENS, \
  ValueSpansFinder

REPORTED_DEPRECATED = {}


class ParserWarnings(Enum):
  org_name_not_found = 1
  org_type_not_found = 2
  org_struct_level_not_found = 3
  date_not_found = 4
  number_not_found = 5
  # 6? what about 6
  value_section_not_found = 7
  contract_value_not_found = 8
  subject_section_not_found = 6  # here it is
  contract_subject_not_found = 9

  protocol_agenda_not_found = 10

  boring_agenda_questions = 11
  contract_subject_section_not_found = 12

  doc_too_big = 13


class Paragraph:
  def __init__(self, header: SemanticTag, body: SemanticTag):
    self.header: SemanticTag = header
    self.body: SemanticTag = body

  def as_combination(self) -> SemanticTag:
    return SemanticTag(self.header.kind + '-' + self.body.kind, None, span=(self.header.span[0], self.body.span[1]))


class LegalDocument:

  def __init__(self, original_text=None, name="legal_doc"):

    self._id = None  # TODO

    self.attributes_tree: DocumentSchema or None = DocumentSchema()
    # self.date: SemanticTag or None = None
    # self.number: SemanticTag or None = None

    self.filename = None
    self._original_text = original_text
    self.warnings: [str] = []

    # todo: use pandas' DataFrame
    self.distances_per_pattern_dict = {}

    self.tokens_map: TextMap or None = None
    self.tokens_map_norm: TextMap or None = None
    self.sentence_map: TextMap or None = None

    self.sections = None  # TODO:deprecated
    self.paragraphs: [Paragraph] = []
    self.name = name

    # subdocs
    self.start = 0
    self.end = None  # TODO:

    self.user: dict = {}

    # TODO: probably we don't have to keep embeddings, just distances_per_pattern_dict
    self.embeddings = None

  def get_id(self):
    return self._id

  def clear_warnings(self):
    self.warnings = []

  def warn(self, msg: ParserWarnings, comment: str = None):
    w = {}
    if comment:
      w['comment'] = comment
    w['code'] = msg.name
    self.warnings.append(w)

  def warn_trimmed(self, maxsize: int):

    appx = f'Для анализа документ обрезан из соображений производительности, допустимая длинна -- {maxsize} слов ✂️'
    wrn = f'{self._id}, {self.filename},  {appx}'
    logger.warning(wrn)

    self.warn(ParserWarnings.doc_too_big, appx)

  def get_headers_as_subdocs(self):
    return [self.subdoc_slice(p.header.as_slice()) for p in self.paragraphs]

  def parse(self, txt=None):
    if txt is None:
      txt = self.original_text

    if txt is None:
      raise ValueError('text is a must')

    _preprocessed_text = self.preprocess_text(txt)
    self.tokens_map = TextMap(_preprocessed_text)
    self.tokens_map_norm = CaseNormalizer().normalize_tokens_map_case(self.tokens_map)

    return self

  def sentence_at_index(self, i: int, return_delimiters=True) -> (int, int):
    # TODO: employ elf.sentence_map
    return self.tokens_map.sentence_at_index(i, return_delimiters)

  def split_into_sentenses(self, sentence_max_len=HyperParameters.protocol_sentence_max_len):
    self.sentence_map = tokenize_doc_into_sentences_map(self.tokens_map.get_full_text(),
                                                        sentence_max_len)

  def __len__(self):
    return self.tokens_map.get_len()

  def __add__(self, suffix: 'LegalDocument'):
    '''
    1) dont forget to add spaces between concatenated docs!!
    2) embeddings are lost
    3)
    :param suffix: doc to add
    :return: self + suffix
    '''
    self.distances_per_pattern_dict = {}
    self._original_text += suffix.original_text

    self.tokens_map += suffix.tokens_map
    self.tokens_map_norm += suffix.tokens_map_norm

    self.sections = None

    self.paragraphs += suffix.paragraphs
    # subdocs
    self.end = suffix.end
    self.embeddings = None

    return self

  def headers_as_sentences(self) -> [str]:
    return headers_as_sentences(self)

  def get_headline(self):
    hh = headers_as_sentences(self)
    if (hh is not None) and len(hh) > 0:
      return hh[0]

  def to_json_obj(self) -> dict:
    j = DocumentJson(self)
    _json_tree = j.__dict__
    _json_tree['attributes_tree'] = {}
    return _json_tree

  def to_json(self) -> str:
    j = DocumentJson(self)
    return json.dumps(j.__dict__, indent=4, ensure_ascii=False, default=lambda o: '<not serializable>')

  def get_tokens_cc(self):
    return self.tokens_map.tokens

  def get_tokens(self):
    return self.tokens_map_norm.tokens

  def get_original_text(self):
    return self._original_text

  def get_normal_text(self) -> str:
    return self.tokens_map.text

  def get_text(self):
    return self.tokens_map.text

  def get_checksum(self):
    return self.tokens_map_norm.get_checksum()

  tokens_cc = property(get_tokens_cc)
  tokens = property(get_tokens)
  original_text = property(get_original_text)
  normal_text = property(get_normal_text, None)
  text = property(get_text)
  checksum = property(get_checksum, None)

  def preprocess_text(self, txt):
    if txt is None:
      raise ValueError('text is a must')
    return normalize_text(txt, replacements_regex)

  def find_sentence_beginnings(self, indices):
    return [find_token_before_index(self.tokens, i, '\n', 0) for i in indices]

  # @profile
  def calculate_distances_per_pattern(self, pattern_factory: AbstractPatternFactory, dist_function=DIST_FUNC,
                                      verbosity=1, merge=False, pattern_prefix=None):
    if self.embeddings is None:
      raise UnboundLocalError(f'Embedd document first, {self._id}')

    self.distances_per_pattern_dict = calculate_distances_per_pattern(self, pattern_factory, dist_function,
                                                                      merge=merge,
                                                                      verbosity=verbosity,
                                                                      pattern_prefix=pattern_prefix)

    return self.distances_per_pattern_dict

  def subdoc_slice(self, __s: slice, name='undef'):

    if self.tokens_map is None:
      raise RuntimeError('self.tokens_map is required, tokenize first')

    # TODO: support None in slice begin
    _s = slice(max((0, __s.start)), max((0, __s.stop)))

    klazz = self.__class__
    sub = klazz(None)
    sub.start = _s.start
    sub.end = _s.stop

    if self.embeddings is not None:
      sub.embeddings = self.embeddings[_s]

    if self.distances_per_pattern_dict is not None:
      sub.distances_per_pattern_dict = {}
      for d in self.distances_per_pattern_dict:
        sub.distances_per_pattern_dict[d] = self.distances_per_pattern_dict[d][_s]

    sub.tokens_map = self.tokens_map.slice(_s)
    sub.tokens_map_norm = self.tokens_map_norm.slice(_s)

    sub.name = f'{self.name}.{name}'
    return sub

  def __getitem__(self, key):
    if isinstance(key, slice):
      # Get the start, stop, and step from the slice
      return self.subdoc_slice(key)
    else:
      raise TypeError("Invalid argument type.")

  def subdoc(self, start, end):
    warnings.warn("use subdoc_slice", DeprecationWarning)
    _s = slice(max(0, start), end)
    return self.subdoc_slice(_s)

  @final
  def embedd_tokens(self, embedder: AbstractEmbedder, max_tokens=8000):
    self.embeddings = embedd_tokens(self.tokens_map_norm,
                                    embedder,
                                    max_tokens=max_tokens,
                                    log_key=f'_id:{self._id}')

  def get_tag_text(self, tag: SemanticTag) -> str:
    return self.tokens_map.text_range(tag.span)

  def substr(self, tag: SemanticTag) -> str:
    return self.tokens_map.text_range(tag.span)


class GenericDocument(LegalDocument):

  def __init__(self, original_text):
    LegalDocument.__init__(self, original_text)
    self.attributes_tree = GenericDocSchema()

  def to_json_obj(self) -> dict:
    j: dict = super().to_json_obj()
    _attributes_tree_dict, _ = to_json(self.attributes_tree)
    j['attributes_tree']["generic"] = _attributes_tree_dict
    return j


class LegalDocumentExt(LegalDocument):

  def __init__(self, doc: LegalDocument):
    super().__init__('')

    if doc is not None:
      # self.__dict__ = doc.__dict__
      self.__dict__.update(doc.__dict__)

    self.sentences_embeddings: Embeddings = None
    self.distances_per_sentence_pattern_dict = {}

  def parse(self, txt=None):
    super().parse(txt)
    self.split_into_sentenses()
    return self

  def subdoc_slice(self, __s: slice, name='undef'):
    sub = super().subdoc_slice(__s, name)
    span = [max((0, __s.start)), max((0, __s.stop))]

    if self.sentence_map:
      sentences_span = self.tokens_map.remap_span(span, self.sentence_map)
      _slice = slice(sentences_span[0], sentences_span[1])
      sub.sentence_map = self.sentence_map.slice(_slice)

      if self.sentences_embeddings is not None:
        sub.sentences_embeddings = self.sentences_embeddings[_slice]
    else:
      raise ValueError(f'split doc into sentences first {self._id}')

    return sub


class DocumentJson:

  @staticmethod
  def from_json_str(json_string: str) -> 'DocumentJson':
    jsondata = json.loads(json_string, object_hook=json_util.object_hook)

    c = DocumentJson(None)
    c.__dict__ = jsondata

    return c

  def __init__(self, doc: LegalDocument):
    self.version = analyser.__version__

    self._id: str = None
    self.original_text = None
    self.normal_text = None
    self.warnings: [str] = []

    self.analyze_timestamp = datetime.datetime.now()
    self.tokenization_maps = {}
    self.size = {}

    if doc is None:
      return

    # ---------------- bred
    self.checksum = doc.get_checksum()
    self.warnings: [str] = list(doc.warnings)

    self.tokenization_maps['words'] = doc.tokens_map.map

    for field in doc.__dict__:
      if field in self.__dict__:
        self.__dict__[field] = doc.__dict__[field]

    self.original_text = doc.original_text
    self.normal_text = doc.normal_text

    self.headers = self.__tags_to_attributes_list([hi.header for hi in doc.paragraphs])

  def __tags_to_attributes_list(self, _tags) -> []:

    attributes = []
    for t in _tags:
      key, attr = t.as_json_attribute()
      attributes.append(attr)

    return attributes

  def dumps(self):
    return json.dumps(self.__dict__, indent=2, ensure_ascii=False, default=json_util.default)


def rectifyed_sum_by_pattern_prefix(distances_per_pattern_dict, prefix, relu_th: float = 0.0):
  warnings.warn("rectifyed_sum_by_pattern_prefix is deprecated", DeprecationWarning)
  vectors = filter_values_by_key_prefix(distances_per_pattern_dict, prefix)
  vectors = [x for x in vectors]
  return rectifyed_sum(vectors, relu_th), len(vectors)


MIN_DOC_LEN = 5


def calculate_distances_per_pattern(doc: LegalDocument, pattern_factory: AbstractPatternFactory,
                                    dist_function=DIST_FUNC, merge=False,
                                    pattern_prefix=None, verbosity=1):
  distances_per_pattern_dict = {}
  if merge:
    distances_per_pattern_dict = doc.distances_per_pattern_dict

  c = 0
  for pat in pattern_factory.patterns:
    if pattern_prefix is None or pat.name[:len(pattern_prefix)] == pattern_prefix:
      if verbosity > 1:
        print(f'estimating distances to pattern {pat.name}', pat)

      dists = make_pattern_attention_vector(pat, doc.embeddings, dist_function)
      distances_per_pattern_dict[pat.name] = dists
      c += 1

  if c == 0:
    raise ValueError('no pattern with prefix: ' + pattern_prefix)

  return distances_per_pattern_dict


def find_value_sign(txt: TextMap) -> (int, (int, int)):
  a = next(txt.finditer(_re_greather_then_1), None)  # не менее, не превышающую
  if a:
    return +1, a

  a = next(txt.finditer(_re_less_then), None)  # менее
  if a:
    return -1, a
  else:
    a = next(txt.finditer(_re_greather_then), None)  # более
    if a:
      return +1, a

  return 0, None


class ContractValue:
  def __init__(self, sign: SemanticTag, value: SemanticTag, currency: SemanticTag, parent: SemanticTag = None):
    warnings.warn("switch to ContractPrice struktur", DeprecationWarning)
    self.value: SemanticTag = value
    self.sign: SemanticTag = sign
    self.currency: SemanticTag = currency
    self.parent: SemanticTag = parent

  def is_child_of(self, p: SemanticTag) -> bool:
    return self.parent.is_child_of(p)

  def as_ContractPrice(self) -> ContractPrice or None:
    warnings.warn("switch to attributes_tree struktur", DeprecationWarning)

    if self.value is None and self.currency is None and self.sign is None:
      return None

    o: ContractPrice = ContractPrice()

    o.amount = clean_semantic_tag_copy(self.value)
    o.currency = clean_semantic_tag_copy(self.currency)
    o.sign = clean_semantic_tag_copy(self.sign)
    confidence = 0.0
    if o.amount is not None:
      confidence = o.amount.confidence

    o.confidence = confidence
    o.span = self.span()

    return o

  def as_list(self) -> [SemanticTag]:
    warnings.warn("switch to attributes_tree struktur", DeprecationWarning)
    if self.sign.value != 0:
      return [self.value, self.sign, self.currency, self.parent]
    else:
      return [self.value, self.currency, self.parent]

  def __add__(self, addon: int) -> 'ContractValue':
    for t in self.as_list():
      t.offset(addon)
    return self

  def span(self):
    left = min([tag.span[0] for tag in self.as_list()])
    right = max([tag.span[1] for tag in self.as_list()])
    return left, right

  def __mul__(self, confidence_k):
    for _r in self.as_list():
      _r.confidence *= confidence_k
    return self

  def integral_sorting_confidence(self) -> float:
    return conditional_p_sum(
      [self.parent.confidence, self.value.confidence, self.currency.confidence, self.sign.confidence])


def extract_sum_sign_currency(doc: LegalDocument, region: (int, int)) -> ContractValue or None:
  subdoc: LegalDocument = doc[region[0] - VALUE_SIGN_MIN_TOKENS: region[1]]

  _sign, _sign_span = find_value_sign(subdoc.tokens_map)

  # ======================================
  try:
    results = ValueSpansFinder(subdoc.text)
  except TypeError:
    results = None
  # ======================================

  if results:
    value_span = subdoc.tokens_map.token_indices_by_char_range(results.number_span)
    currency_span = subdoc.tokens_map.token_indices_by_char_range(results.currency_span)

    group = SemanticTag('sign_value_currency', value=None, span=region)

    sign = SemanticTag(ContractTags.Sign.display_string, _sign, _sign_span, parent=group)
    sign.offset(subdoc.start)

    value_tag = SemanticTag(ContractTags.Value.display_string, results.value, value_span, parent=group)
    value_tag.offset(subdoc.start)

    currency = SemanticTag(ContractTags.Currency.display_string, results.currencly_name, currency_span,
                           parent=group)
    currency.offset(subdoc.start)

    groupspan = [0, 0]
    groupspan[0] = min(sign.span[0], value_tag.span[0], currency.span[0], group.span[0])
    groupspan[1] = max(sign.span[1], value_tag.span[1], currency.span[1], group.span[1])
    group.span = groupspan

    # TODO: return ContractPrice
    return ContractValue(sign, value_tag, currency, group)
  else:
    return None


def tokenize_doc_into_sentences_map(txt: str, max_len_chars=150) -> TextMap:
  tm = TextMap('', [])

  # body_lines = doc.tokens_map._full_text.splitlines(True)
  body_lines = txt.splitlines(True)
  for line in body_lines:
    tm += split_sentences_into_map(line, max_len_chars)

  return tm


PARAGRAPH_DELIMITER = '\n'


def embedd_sentences(text_map: TextMap, embedder: AbstractEmbedder, log_addon='',
                     max_tokens=HyperParameters.max_sentenses_to_embedd):
  warnings.warn("use embedd_words", DeprecationWarning)
  logger.info(f'{log_addon} {len(text_map)}')
  if text_map is None:
    # https://github.com/nemoware/analyser/issues/224
    raise ValueError('text_map must not be None')

  if len(text_map) > max_tokens:
    return embedder.embedd_large(text_map, max_tokens, log_addon)
  else:
    return embedder.embedd_tokens(text_map.tokens)


def make_headline_attention_vector(doc):
  parser_headline_attention_vector = np.zeros(len(doc))

  for p in doc.paragraphs:
    parser_headline_attention_vector[p.header.slice] = 1

  return parser_headline_attention_vector


def headers_as_sentences(doc: LegalDocument, normal_case=True, strip_number=True) -> [str]:
  _map = doc.tokens_map
  if normal_case:
    _map = doc.tokens_map_norm

  numbered = [_map.slice(p.header.as_slice()) for p in doc.paragraphs]
  stripped: [str] = []

  for s in numbered:
    if strip_number:
      a = get_tokenized_line_number(s.tokens, 0)
      _, span, _, _ = a
      line = s.text_range([span[1], None]).strip()
    else:
      line = s.text
    stripped.append(line)

  return stripped


def remap_attention_vector(v: FixedVector, source_map: TextMap, target_map: TextMap) -> FixedVector:
  av = np.zeros(len(target_map))

  for i in range(len(source_map)):
    span = i, i + 1

    t_span = source_map.remap_span(span, target_map)
    av[t_span[0]:t_span[1]] = v[i]
  return av


def embedd_tokens(tokens_map_norm: TextMap, embedder: AbstractEmbedder, max_tokens=8000, log_key=''):
  ch = tokens_map_norm.get_checksum()

  _cached = embedder.get_cached_embedding(ch)
  if _cached is not None:
    logger.debug(f'getting embedding from cache {log_key}')
    return _cached
  else:
    logger.info(f'embedding doc {log_key}')
    if tokens_map_norm.tokens:

      if len(tokens_map_norm) > max_tokens:
        embeddings = embedder.embedd_large(tokens_map_norm, max_tokens, log_key)
      else:
        embeddings = embedder.embedd_tokens(tokens_map_norm.tokens)

      embedder.cache_embedding(ch, embeddings)
      return embeddings
    else:
      raise ValueError(f'cannot embedd doc {log_key}, no tokens')
