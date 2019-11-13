#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8


# legal_docs.py
import datetime
import gc
import json
from functools import wraps

from bson import json_util

from documents import TextMap
from embedding_tools import AbstractEmbedder
from ml_tools import normalize, smooth_safe, max_exclusive_pattern, SemanticTag, conditional_p_sum, put_if_better
from patterns import *
from structures import ORG_2_ORG, ContractTags
from tests.test_text_tools import split_sentences_into_map
from text_normalize import *
from text_tools import *
from transaction_values import _re_greather_then, _re_less_then, _re_greather_then_1, VALUE_SIGN_MIN_TOKENS, \
  find_value_spans

REPORTED_DEPRECATED = {}


def remove_sr_duplicates_conditionally(list_: PatternSearchResults):
  ret = []
  dups = {}
  for r in list_:
    put_if_better(dups, r.key_index, r, lambda a, b: a.confidence > b.confidence)

  for x in dups.values():
    ret.append(x)
  del dups
  return ret


def substract_search_results(a: PatternSearchResults, b: PatternSearchResults) -> PatternSearchResults:
  b_indexes = [x.key_index for x in b]
  result: PatternSearchResults = []
  for x in a:
    if x.key_index not in b_indexes:
      result.append(x)
  return result


def deprecated(fn):
  @wraps(fn)
  @wraps(fn)
  def with_reporting(*args, **kwargs):
    if fn.__name__ not in REPORTED_DEPRECATED:
      REPORTED_DEPRECATED[fn.__name__] = 1
      print("----WARNING!: function {} is deprecated".format(fn.__name__))

    ret = fn(*args, **kwargs)
    return ret

  return with_reporting


class Paragraph:
  def __init__(self, header: SemanticTag, body: SemanticTag):
    self.header: SemanticTag = header
    self.body: SemanticTag = body


class LegalDocument:

  def __init__(self, original_text=None, name="legal_doc"):

    self._id = None  # TODO
    self.date = None

    self.filename = None
    self._original_text = original_text
    self._normal_text = None

    # todo: use pandas' DataFrame
    self.distances_per_pattern_dict = {}

    self.tokens_map: TextMap = None
    self.tokens_map_norm: TextMap or None = None

    self.sections = None  # TODO:deprecated
    self.paragraphs: List[Paragraph] = []
    self.name = name

    # subdocs
    self.start = 0
    self.end = None  # TODO:

    # TODO: probably we don't have to keep embeddings, just distances_per_pattern_dict
    self.embeddings = None

  def parse(self, txt=None):
    if txt is None:
      txt = self.original_text

    assert txt is not None

    self._normal_text = self.preprocess_text(txt)
    self.tokens_map = TextMap(self._normal_text)

    self.tokens_map_norm = CaseNormalizer().normalize_tokens_map_case(self.tokens_map)
    # body = SemanticTag(kind=None, value=None, span=(0, len(self.tokens_map)));
    # header = SemanticTag(kind=None, value=None, span=(0, 0));
    # self.paragraphs = [Paragraph(header, body)]
    return self

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
    assert self._normal_text is not None
    assert suffix._normal_text is not None

    self.distances_per_pattern_dict = {}
    self._normal_text += suffix.normal_text
    self._original_text += suffix.original_text

    self.tokens_map += suffix.tokens_map
    self.tokens_map_norm += suffix.tokens_map_norm

    self.sections = None

    self.paragraphs += suffix.paragraphs
    # subdocs
    self.end = suffix.end
    self.embeddings = None

    return self

  def get_tags(self) -> [SemanticTag]:
    return []

  def get_tags_attention(self):
    _attention = np.zeros(self.__len__())

    for t in self.get_tags():
      _attention[t.as_slice()] += t.confidence
    return _attention

  def to_json(self) -> str:
    j = DocumentJson(self)
    return json.dumps(j.__dict__, indent=4, ensure_ascii=False, default=lambda o: '<not serializable>')

  def get_tokens_cc(self):
    return self.tokens_map.tokens

  def get_tokens(self):
    return self.tokens_map_norm.tokens

  def get_original_text(self):
    return self._original_text

  def get_normal_text(self):
    return self._normal_text

  def get_text(self):
    return self.tokens_map.text

  tokens_cc = property(get_tokens_cc)
  tokens = property(get_tokens)
  original_text = property(get_original_text)
  normal_text = property(get_normal_text)
  text = property(get_text)

  def preprocess_text(self, txt):
    assert txt is not None
    return normalize_text(txt, replacements_regex)

  def find_sentence_beginnings(self, indices):
    return [find_token_before_index(self.tokens, i, '\n', 0) for i in indices]

  # @profile
  def calculate_distances_per_pattern(self, pattern_factory: AbstractPatternFactory, dist_function=DIST_FUNC,
                                      verbosity=1, merge=False, pattern_prefix=None):
    assert self.embeddings is not None
    self.distances_per_pattern_dict = calculate_distances_per_pattern(self, pattern_factory, dist_function, merge=merge,
                                                                      verbosity=verbosity,
                                                                      pattern_prefix=pattern_prefix)

    return self.distances_per_pattern_dict

  def subdoc_slice(self, __s: slice, name='undef'):
    assert self.tokens_map is not None
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

  @deprecated
  def subdoc(self, start, end):
    warnings.warn("use subdoc_slice", DeprecationWarning)
    _s = slice(max(0, start), end)
    return self.subdoc_slice(_s)

  def make_attention_vector(self, factory, pattern_prefix, recalc_distances=True) -> (List[float], str):
    # ---takes time
    if recalc_distances:
      calculate_distances_per_pattern(self, factory, merge=True, pattern_prefix=pattern_prefix)
    # ---
    vectors = filter_values_by_key_prefix(self.distances_per_pattern_dict, pattern_prefix)
    vectors_i = []

    attention_vector_name = AV_PREFIX + pattern_prefix
    attention_vector_name_soft = AV_SOFT + attention_vector_name

    for v in vectors:
      if max(v) > 0.6:  # TODO: get rid of magic numbers
        vector_i, _ = improve_attention_vector(self.embeddings, v, relu_th=0.6, mix=0.9)
        vectors_i.append(vector_i)
      else:
        vectors_i.append(v)

    x = max_exclusive_pattern(vectors_i)
    self.distances_per_pattern_dict[attention_vector_name_soft] = x
    x = relu(x, 0.8)

    self.distances_per_pattern_dict[attention_vector_name] = x
    return x, attention_vector_name

  def find_sentences_by_pattern_prefix(self, org_level, factory, pattern_prefix) -> PatternSearchResults:

    """
    :param factory:
    :param pattern_prefix:
    :return:
    """
    warnings.warn("use find_sentences_by_attention_vector", DeprecationWarning)
    attention, attention_vector_name = self.make_attention_vector(factory, pattern_prefix)

    results: PatternSearchResults = []

    for i in np.nonzero(attention)[0]:
      _b = self.tokens_map.sentence_at_index(i)
      _slice = slice(_b[0], _b[1])

      if _slice.stop != _slice.start:

        sum_ = sum(attention[_slice])
        #       confidence = np.mean( np.nonzero(x[sl]) )
        nonzeros_count = len(np.nonzero(attention[_slice])[0])
        confidence = 0

        if nonzeros_count > 0:
          confidence = sum_ / nonzeros_count

        if confidence > 0.8:
          r = PatternSearchResult(ORG_2_ORG[org_level], _slice)
          r.attention_vector_name = attention_vector_name
          r.pattern_prefix = pattern_prefix
          r.confidence = confidence
          r.parent = self

          results.append(r)

    results = remove_sr_duplicates_conditionally(results)

    return results

  def reset_embeddings(self):
    print('-----ARE YOU SURE YOU NEED TO DROP EMBEDDINGS NOW??---------')
    del self.embeddings
    self.embeddings = None
    gc.collect()

  @deprecated
  def embedd(self, pattern_factory):
    warnings.warn("use embedd_tokens, provide embedder", DeprecationWarning)
    self.embedd_tokens(pattern_factory.embedder)

  def embedd_tokens(self, embedder: AbstractEmbedder, verbosity=2):
    if self.tokens:
      max_tokens = 7000
      if len(self.tokens_map_norm) > max_tokens:
        self._embedd_large(embedder, max_tokens, verbosity)
      else:
        self.embeddings = self._emb(self.tokens, embedder)
    else:
      raise ValueError(f'cannot embed doc {self.filename}, no tokens')

  # @profile
  def _emb(self, tokens, embedder):
    embeddings, _g = embedder.embedd_tokenized_text([tokens], [len(tokens)])
    embeddings = embeddings[0]
    return embeddings

  def _embedd_large(self, embedder, max_tokens=8000, verbosity=2):

    overlap = 100  # max_tokens // 5

    number_of_windows = 1 + len(self.tokens_map_norm) // max_tokens
    window = max_tokens

    if verbosity > 1:
      print(
        "WARNING: Document is too large for embedding: {} tokens. Splitting into {} windows overlapping with {} tokens ".format(
          len(self.tokens_map_norm), number_of_windows, overlap))

    start = 0
    embeddings = None
    # tokens = []
    while start < len(self.tokens_map_norm):

      subtokens = self.tokens_map_norm[start:start + window + overlap]
      if verbosity > 2:
        print("Embedding region:", start, len(subtokens))

      sub_embeddings = self._emb(subtokens, embedder)[0:window]

      # sub_embeddings = sub_embeddings[0:window]
      # subtokens = subtokens[0:window]

      if embeddings is None:
        embeddings = sub_embeddings
      else:
        embeddings = np.concatenate([embeddings, sub_embeddings])
      # tokens += subtokens

      start += window

    self.embeddings = embeddings
    # self.tokens = tokens

  def get_tag_text(self, tag: SemanticTag):
    return self.tokens_map.text_range(tag.span)

  def substr(self, tag: SemanticTag) -> str:
    return self.tokens_map.text_range(tag.span)

  def tag_value(self, tagname):
    t = SemanticTag.find_by_kind(self.get_tags(), tagname)
    if t:
      return t.value
    else:
      return None


class DocumentJson:

  @staticmethod
  def from_json(json_string: str) -> 'DocumentJson':
    jsondata = json.loads(json_string, object_hook=json_util.object_hook)

    c = DocumentJson(None)
    c.__dict__ = jsondata

    return c

  def __init__(self, doc: LegalDocument):

    self._id: str = None
    self.original_text = None
    self.normal_text = None

    self.analyze_timestamp = datetime.datetime.now()
    self.tokenization_maps = {}

    if doc is None:
      return

    self.checksum = hash(doc.normal_text)
    self.tokenization_maps['words'] = doc.tokens_map.map

    for field in doc.__dict__:
      if field in self.__dict__:
        self.__dict__[field] = doc.__dict__[field]

    self.original_text = doc.original_text
    self.normal_text = doc.normal_text

    self.attributes = self.__tags_to_attributes_dict(doc.get_tags())
    self.headers = self.__tags_to_attributes_list([hi.header for hi in doc.paragraphs])

  def __tags_to_attributes_list(self, _tags):

    attributes = []
    for t in _tags:
      val = t.__dict__.copy()
      attributes.append(val)
      del val['kind']
    return attributes

  def __tags_to_attributes_dict(self, _tags):

    cnt = 0
    attributes = {}
    for t in _tags:
      cnt += 1
      key = t.kind.replace('.', '_')
      if key in attributes:
        key = f'{key}_{cnt}'
      attributes[key] = t.__dict__.copy()
      del attributes[key]['kind']

    return attributes

  def dumps(self):
    return json.dumps(self.__dict__, indent=2, ensure_ascii=False, default=json_util.default)


def rectifyed_sum_by_pattern_prefix(distances_per_pattern_dict, prefix, relu_th: float = 0.0):
  warnings.warn("rectifyed_sum_by_pattern_prefix is deprecated", DeprecationWarning)
  vectors = filter_values_by_key_prefix(distances_per_pattern_dict, prefix)
  vectors = [x for x in vectors]
  return rectifyed_sum(vectors, relu_th), len(vectors)


def mean_by_pattern_prefix(distances_per_pattern_dict, prefix):
  warnings.warn("deprecated", DeprecationWarning)
  #     print('mean_by_pattern_prefix', prefix, relu_th)
  _sum, c = rectifyed_sum_by_pattern_prefix(distances_per_pattern_dict, prefix, relu_th=0.0)
  return normalize(_sum)


def rectifyed_normalized_mean_by_pattern_prefix(distances_per_pattern_dict, prefix, relu_th=0.5):
  return normalize(rectifyed_mean_by_pattern_prefix(distances_per_pattern_dict, prefix, relu_th))


def rectifyed_mean_by_pattern_prefix(distances_per_pattern_dict, prefix, relu_th=0.5):
  #     print('mean_by_pattern_prefix', prefix, relu_th)
  _sum, c = rectifyed_sum_by_pattern_prefix(distances_per_pattern_dict, prefix, relu_th)
  _sum /= c
  return _sum


class BasicContractDocument(LegalDocument):

  def __init__(self, original_text=None):
    LegalDocument.__init__(self, original_text)

  def get_subject_ranges(self, indexes_zipped, section_indexes: List):

    subj_range = None
    head_range = None
    for i in range(len(indexes_zipped) - 1):
      if indexes_zipped[i][0] == 1:
        subj_range = range(indexes_zipped[i][1], indexes_zipped[i + 1][1])
      if indexes_zipped[i][0] == 0:
        head_range = range(indexes_zipped[i][1], indexes_zipped[i + 1][1])
    if head_range is None:
      print("WARNING: Contract type might be not known!!")
      head_range = range(0, 0)
    if subj_range is None:
      print("WARNING: Contract subject might be not known!!")
      if len(self.tokens) < 80:
        _end = len(self.tokens)
      else:
        _end = 80
      subj_range = range(0, _end)
    return head_range, subj_range


# SUMS -----------------------------
ProtocolDocument = LegalDocument


# Charter Docs


class CharterDocument(LegalDocument):
  def __init__(self, original_text, name="charter"):
    LegalDocument.__init__(self, original_text, name)

    self._constraints: List[PatternSearchResult] = []
    self.value_constraints = {}

    self._org = None

    self.org_type_tag: SemanticTag = None
    self.org_name_tag: SemanticTag = None

    # TODO:remove it
    self._charity_constraints_old = {}
    self._value_constraints_old = {}

  def get_tags(self) -> [SemanticTag]:
    return [self.org_type_tag, self.org_name_tag]

  def get_org(self):
    warnings.warn("use org_type_tag and org_name_tag", DeprecationWarning)
    return self._org

  def set_org(self, org):
    warnings.warn("use org_type_tag and org_name_tag", DeprecationWarning)
    self._org = org

  def get_constraints_old(self):
    return self._value_constraints_old

  constraints_old = property(get_constraints_old)
  org = property(get_org, set_org)

  def constraints_by_org_level(self, org_level: OrgStructuralLevel, constraint_subj: ContractSubject = None) -> List[
    PatternSearchResult]:
    for p in self._constraints:
      if p.org_level is org_level and (constraint_subj is None or p.subject_mapping['subj'] == constraint_subj):
        yield p


def max_by_pattern_prefix(distances_per_pattern_dict, prefix, attention_vector=None):
  ret = {}

  for p in distances_per_pattern_dict:
    if p.startswith(prefix):
      x = distances_per_pattern_dict[p]

      if attention_vector is not None:
        x = np.array(x)
        x += attention_vector

      ret[p] = x.argmax()

  return ret


def split_into_sections(doc, caption_indexes):
  sorted_keys = sorted(caption_indexes, key=lambda s: caption_indexes[s])

  doc.subdocs = []
  for i in range(1, len(sorted_keys)):
    key = sorted_keys[i - 1]
    next_key = sorted_keys[i]
    start = caption_indexes[key]
    end = caption_indexes[next_key]
    # print(key, [start, end])

    subdoc = doc.subdoc(start, end)
    subdoc.filename = key
    doc.subdocs.append(subdoc)


MIN_DOC_LEN = 5


@deprecated
def make_soft_attention_vector(doc: LegalDocument, pattern_prefix, relu_th=0.5, blur=60, norm=True):
  warnings.warn("make_soft_attention_vector is deprecated", DeprecationWarning)
  assert doc.distances_per_pattern_dict is not None

  if len(doc.tokens) < MIN_DOC_LEN:
    print("----ERROR: make_soft_attention_vector: too few tokens {} ".format(doc.text))
    return np.full(len(doc.tokens), 0.0001)

  attention_vector, _c = rectifyed_sum_by_pattern_prefix(doc.distances_per_pattern_dict, pattern_prefix,
                                                         relu_th=relu_th)
  attention_vector = relu(attention_vector, relu_th=relu_th)

  attention_vector = smooth_safe(attention_vector, window_len=blur)
  attention_vector = smooth_safe(attention_vector, window_len=blur)
  try:
    if norm:
      attention_vector = normalize(attention_vector)
  except:
    print(
      "----ERROR: make_soft_attention_vector: attention_vector for pattern prefix {} is not contrast, len = {}".format(
        pattern_prefix, len(attention_vector)))
    attention_vector = np.full(len(attention_vector), attention_vector[0])

  return attention_vector


@deprecated
def soft_attention_vector(doc, pattern_prefix, relu_th=0.5, blur=60, norm=True):
  warnings.warn("deprecated", DeprecationWarning)
  assert doc.distances_per_pattern_dict is not None

  if len(doc.tokens) < MIN_DOC_LEN:
    print("----ERROR: soft_attention_vector: too few tokens {} ".format(doc.text))
    return np.full(len(doc.tokens), 0.0001)

  attention_vector, c = rectifyed_sum_by_pattern_prefix(doc.distances_per_pattern_dict, pattern_prefix, relu_th=relu_th)
  assert c > 0
  attention_vector = relu(attention_vector, relu_th=relu_th)

  attention_vector = smooth_safe(attention_vector, window_len=blur)
  attention_vector = smooth_safe(attention_vector, window_len=blur)
  attention_vector /= c
  try:
    if norm:
      attention_vector = normalize(attention_vector)
  except:
    print("----ERROR: soft_attention_vector: attention_vector for pattern prefix {} is not contrast, len = {}".format(
      pattern_prefix, len(attention_vector)))

    attention_vector = np.full(len(attention_vector), attention_vector[0])
  return attention_vector


def _expand_slice(s: slice, exp):
  return slice(s.start - exp, s.stop + exp)


def calculate_distances_per_pattern(doc: LegalDocument, pattern_factory: AbstractPatternFactory,
                                    dist_function=DIST_FUNC, merge=False,
                                    pattern_prefix=None, verbosity=1):
  distances_per_pattern_dict = {}
  if merge:
    distances_per_pattern_dict = doc.distances_per_pattern_dict

  c = 0
  for pat in pattern_factory.patterns:
    if pattern_prefix is None or pat.name[:len(pattern_prefix)] == pattern_prefix:
      if verbosity > 1: print(f'estimating distances to pattern {pat.name}', pat)

      dists = make_pattern_attention_vector(pat, doc.embeddings, dist_function)
      distances_per_pattern_dict[pat.name] = dists
      c += 1

  # if verbosity > 0:
  #   print(distances_per_pattern_dict.keys())
  if c == 0:
    raise ValueError('no pattern with prefix: ' + pattern_prefix)

  return distances_per_pattern_dict


def detect_sign_2(txt: TextMap) -> (int, (int, int)):
  """
  todo: rename to 'find_value_sign'
  :param txt:
  :return:
  """

  a = next(txt.finditer(_re_greather_then_1), None)  # не менее, не превышающую
  if a:
    return +1, a

  a = next(txt.finditer(_re_less_then), None)  # менее
  if a:
    return -1, a
  else:
    a = next(txt.finditer(_re_greather_then), None)
    if a:
      return +1, a

  return 0, None


find_value_sign = detect_sign_2


class ContractValue:
  def __init__(self, sign: SemanticTag, value: SemanticTag, currency: SemanticTag, parent: SemanticTag = None):
    self.value = value
    self.sign = sign
    self.currency = currency
    self.parent = parent

  def as_list(self) -> [SemanticTag]:
    return [self.value, self.sign, self.currency, self.parent]

  def span(self):
    left = min([tag.span[0] for tag in self.as_list()])
    right = max([tag.span[0] for tag in self.as_list()])
    return left, right

  def integral_sorting_confidence(self) -> float:
    return conditional_p_sum(
      [self.parent.confidence, self.value.confidence, self.currency.confidence, self.sign.confidence])


def extract_sum_sign_currency(doc: LegalDocument, region: (int, int)) ->  ContractValue or None:
  subdoc: LegalDocument = doc[region[0] - VALUE_SIGN_MIN_TOKENS: region[1]]

  _sign, _sign_span = find_value_sign(subdoc.tokens_map)

  # ======================================
  results = find_value_spans(subdoc.text)
  # ======================================

  if results:
    value_char_span, value, currency_char_span, currency = results
    value_span = subdoc.tokens_map.token_indices_by_char_range_2(value_char_span)
    currency_span = subdoc.tokens_map.token_indices_by_char_range_2(currency_char_span)

    parent = f'sign_value_currency_{region[0]}_{region[1]}'
    group = SemanticTag(parent, None, region)

    sign = SemanticTag(ContractTags.Sign.display_string, _sign, _sign_span)
    sign.parent = parent
    sign.offset(subdoc.start)

    value_tag = SemanticTag(ContractTags.Value.display_string, value, value_span)
    value_tag.parent = parent
    value_tag.offset(subdoc.start)

    currency = SemanticTag(ContractTags.Currency.display_string, currency, currency_span)
    currency.parent = parent
    currency.offset(subdoc.start)

    return ContractValue(sign, value_tag, currency, group)
  else:
    return None


#
#
# if __name__ == '__main__':
#
#   ex="""
#   с учетом положений пункта 8.4.4 устава , одобрение заключения , изменения , продления , возобновления или расторжения обществом ( i ) любых договоров страхования , если стоимость соответствующего договора или нескольких взаимосвязанных договоров превышает 100 000 ( сто тысяч ) долларов сша ( или эквивалент этой суммы в рублях или иной валюте ) , либо ( ii ) договоров страхования , относящихся к операторскому договору 6к или связанных с операторским договором 6к до закрытия сделки 6к независимо от суммы ;
#   """
#   #
#   # self.pattern_prefix: str = None
#   # self.attention_vector_name: str = None
#   # self.parent: LegalDocument = None
#   # self.confidence: float = 0
#   # self.region: slice = None
#
#   def extract_sum_and_sign_3(tokens:Tokens, region: slice) -> ValueConstraint:
#     _slice = slice(region.start - VALUE_SIGN_MIN_TOKENS, region.stop)
#     subtokens = tokens[_slice]
#     _prefix_tokens = subtokens[0:VALUE_SIGN_MIN_TOKENS + 1]
#     _prefix = untokenize(_prefix_tokens)
#     _sign = detect_sign(_prefix)
#     # ======================================
#     _sum = extract_sum_from_tokens_2(subtokens)
#     # ======================================
#
#     currency = "UNDEF"
#     value = np.nan
#     if _sum is not None:
#       currency = _sum[1]
#       if _sum[1] in currencly_map:
#         currency = currencly_map[_sum[1]]
#       value = _sum[0]
#
#     vc = ValueConstraint(value, currency, _sign, TokensWithAttention([], []))
#
#     return vc
#
#
#   def extract_all_contraints_from_sr( sentence ) :
#
#     tokens = sentence.split(' ')
#     def tokens_before_index(string, index):
#       return len(string[:index].split(' '))
#
#
#     all = [slice(m.start(0), m.end(0)) for m in re.finditer(complete_re, sentence)]
#     constraints: List[ProbableValue] = []
#
#     for a in all:
#       print(tokens_before_index(sentence, a.start), 'from', sentence[a])
#       token_index_s = tokens_before_index(sentence, a.start) - 1
#       token_index_e = tokens_before_index(sentence, a.stop)
#
#       region = slice(token_index_s, token_index_e)
#       print('region=', region)
#
#
#       vc = extract_sum_and_sign_3(tokens, region)
#       print(vc.sign)
#       # _e = _expand_slice(region, 10)
#       # vc.context = TokensWithAttention(search_result.tokens[_e], attention_vector[_e])
#       # confidence = attention_vector[region.start]
#       # pv = ProbableValue(vc, confidence)
#       #
#       # constraints.append(pv)
#
#     # return constraints
#
#
#
#   extract_all_contraints_from_sr(ex)


def tokenize_doc_into_sentences_map(doc: LegalDocument, max_len_chars=150) -> TextMap:
  tm = TextMap('', [])

  # if doc.paragraphs:
  #   for p in doc.paragraphs:
  #     header_lines = doc.substr(p.header).splitlines(True)
  #     for line in header_lines:
  #       tm += split_sentences_into_map(line, max_len_chars)
  #
  #     body_lines = doc.substr(p.body).splitlines(True)
  #     for line in body_lines:
  #       tm += split_sentences_into_map(line, max_len_chars)
  # else:
  body_lines = doc.text.splitlines(True)
  for line in body_lines:
    tm += split_sentences_into_map(line, max_len_chars)

  return tm


PARAGRAPH_DELIMITER = '\n'
