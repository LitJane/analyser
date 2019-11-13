# origin: charter_parser.py
import re

from charter_patterns import make_constraints_attention_vectors
from legal_docs import LegalDocument, CharterDocument, \
  _expand_slice
from ml_tools import *
from ml_tools import FixedVector, ProbableValue
from parsing import ParsingSimpleContext, head_types_dict, known_subjects
from patterns import find_ner_end, improve_attention_vector, AV_PREFIX, PatternSearchResult, \
  ConstraintsSearchResult, PatternSearchResults, PatternMatch
from sections_finder import SectionsFinder, FocusingSectionsFinder, HeadlineMeta
from structures import *
from text_tools import untokenize, Tokens
from transaction_values import extract_sum, number_re, ValueConstraint, VALUE_SIGN_MIN_TOKENS, detect_sign, \
  currencly_map, complete_re
from violations import ViolationsFinder

WARN = '\033[1;31m'


class CharterConstraintsParser(ParsingSimpleContext):

  def __init__(self, pattern_factory):
    ParsingSimpleContext.__init__(self)
    self.pattern_factory = pattern_factory
    pass

  # ---------------------------------------
  def extract_constraint_values_from_sections(self, sections):
    rez = {}

    for head_type in sections:
      section = sections[head_type]
      rez[head_type] = self.extract_constraint_values_from_section(section)

    return rez

  # ---------------------------------------
  def extract_constraint_values_from_section(self, section: HeadlineMeta):

    if self.verbosity_level > 1:
      print('extract_constraint_values_from_section', section.type)

    body = section.body

    body.calculate_distances_per_pattern(self.pattern_factory, pattern_prefix='sum_max', merge=True)
    body.calculate_distances_per_pattern(self.pattern_factory, pattern_prefix='sum__', merge=True)
    body.calculate_distances_per_pattern(self.pattern_factory, pattern_prefix='d_order_', merge=True)

    a_vectors = make_constraints_attention_vectors(body)
    body.distances_per_pattern_dict = {**body.distances_per_pattern_dict, **a_vectors}

    if self.verbosity_level > 1:
      print('extract_constraint_values_from_section', 'embedding....')

    sentenses_having_values: List[LegalDocument] = []
    # senetences = split_by_token(body.tokens, '\n')

    for _slice in body.tokens_map_norm.split_spans('\n'):

      __line = body.tokens_map_norm.text_range(_slice)
      _sum = extract_sum(__line)

      if _sum is not None:
        ss_subdoc = body.subdoc_slice(_slice, name=f'value_sent:{_slice.start}')
        sentenses_having_values.append(ss_subdoc)

      if self.verbosity_level > 2:
        print('-', _sum, __line)

    r_by_head_type = {
      'section': head_types_dict[section.type],
      'caption': section.subdoc.tokens_map.text,
      'sentences': self.__extract_constraint_values_from_region(sentenses_having_values)
    }
    self._logstep(f"Finding margin transaction values in section {section.subdoc.tokens_map.text}")
    return r_by_head_type

  ##---------------------------------------
  @staticmethod
  def __extract_constraint_values_from_region(sentenses_i: List[LegalDocument]):
    warnings.warn("deprecated: this method must be rewritten completely", DeprecationWarning)
    if sentenses_i is None or len(sentenses_i) == 0:
      return []

    sentences = []
    for sentence_subdoc in sentenses_i:
      constraints: List[ProbableValue] = extract_all_contraints_from_sentence(sentence_subdoc,
                                                                              sentence_subdoc.distances_per_pattern_dict[
                                                                                'deal_value_attention_vector'])

      sentence = {
        'subdoc': sentence_subdoc,
        'constraints': constraints
      }

      sentences.append(sentence)
    return sentences


def extract_all_contraints_from_sentence(sentence_subdoc: LegalDocument, attention_vector: List[float]) -> List[
  ProbableValue]:
  warnings.warn("deprecated", DeprecationWarning)
  tokens = sentence_subdoc.tokens
  assert len(attention_vector) == len(tokens)

  text_fragments, indexes, ranges = split_by_number_2(tokens, attention_vector, 0.2)

  constraints: List[ProbableValue] = []
  if len(indexes) > 0:

    for region in ranges:
      vc = extract_sum_and_sign_2(sentence_subdoc, region)

      _e = _expand_slice(region, 10)
      vc.context = TokensWithAttention(tokens[_e], attention_vector[_e])
      confidence = attention_vector[region.start]
      pv = ProbableValue(vc, confidence)

      constraints.append(pv)

  return constraints


""" ❤️ == GOOD CharterDocumentParser  ====================================== """


class CharterDocumentParser(CharterConstraintsParser):

  def __init__(self, pattern_factory):
    CharterConstraintsParser.__init__(self, pattern_factory)

    self.sections_finder: SectionsFinder = FocusingSectionsFinder(self)

    self.doc: CharterDocument = None

    self.violations_finder = ViolationsFinder()

  def analyze_charter(self, txt, verbosity=2):
    """
    🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀
    :param txt:
    """

    self._reset_context()

    # 0. parse
    _charter_doc = CharterDocument(txt)

    # 1. find top level structure
    _charter_doc.parse()
    _charter_doc.embedd_tokens(self.pattern_factory.embedder, verbosity)
    self.doc: CharterDocument = _charter_doc

    """ 2. ✂️ 📃 -> 📄📄📄  finding headlines (& sections) ==== ️"""

    competence_v = self._make_competence_attention_v()
    self.sections_finder.find_sections(self.doc, self.pattern_factory, self.pattern_factory.headlines,
                                       headline_patterns_prefix='headline.', additional_attention=competence_v)

    """ 2. NERS 🏦 🏨 🏛==== ️"""
    _org_, self.charter.org_type_tag, self.charter.org_name_tag = self.ners()
    self.charter.org = _org_  # TODO: remove it, this is just for compatibility

    """ 3. CONSTRAINTS 💰 💵 ==== ️"""
    self.find_contraints_2()

    ##----end, logging, closing
    self.verbosity_level = 1
    self.log_warnings()

    return self.org, self.constraints

  def _make_competence_attention_v(self):
    self.doc.calculate_distances_per_pattern(self.pattern_factory, pattern_prefix='competence', merge=True)
    filtered = filter_values_by_key_prefix(self.doc.distances_per_pattern_dict, 'competence')

    competence_v = rectifyed_sum(filtered, 0.3)
    competence_v, c = improve_attention_vector(self.doc.embeddings, competence_v, mix=1)
    return competence_v

  def ners(self):
    """
    org is depreceated, use org_type_tag and org_name_tag only!!
    :return:
    """
    if 'name' in self.doc.sections:
      section: HeadlineMeta = self.doc.sections['name']
      org, org_type_tag, org_name_tag = self.detect_ners(section.body)

    else:
      self.warning('Секция наименования компнании не найдена')
      self.warning('Попытаемся искать просто в начале документа 🚑')
      org, org_type_tag, org_name_tag = self.detect_ners(self.doc.subdoc_slice(slice(0, 3000), name='name_section'))

    self._logstep("extracting NERs (named entities 🏦 🏨 🏛)")

    """ 🚀️ = 🍄 🍄 🍄 🍄 🍄   TODO: ============================ """
    # todo: do not return org
    return org, org_type_tag, org_name_tag

  """ 🚀️ == GOOD CharterDocumentParser  ====================================================== """

  def parse(self, doc: CharterDocument):
    self.doc: CharterDocument = doc

    # TODO: move to doc.dict
    self.deal_attention = None  # make_improved_attention_vector(self.doc, 'd_order_')
    # 💵 💵 💰
    # TODO: move to doc.dict
    self.value_attention = None  # make_improved_attention_vector(self.doc, 'sum__')

  def _get_head_sections(self, prefix='head.'):
    sections_filtered = {}

    for k in self.doc.sections:
      if k[:len(prefix)] == prefix:
        sections_filtered[k] = self.doc.sections[k]
    return sections_filtered

  """
  🚷🔥
  """

  def find_contraints_2(self) -> None:

    # 5. extract constraint values
    sections_filtered = self._get_head_sections()

    for section_name in sections_filtered:
      section = sections_filtered[section_name].body

      value_constraints, charity_constraints, all_margin_values, charity_constraints = self._find_constraints_in_section(
        org_level=section_name, section=section)

      self.charter._charity_constraints_old[section_name] = charity_constraints
      self.charter._value_constraints_old[section_name] = value_constraints

      self.charter._constraints += charity_constraints
      self.charter._constraints += all_margin_values

  def _find_constraints_in_section(self, org_level: str, section):
    for subj in known_subjects:
      pattern_prefix = f'x_{subj}'
      attention, attention_vector_name = section.make_attention_vector(self.pattern_factory, pattern_prefix)

    # TODO: try 'margin_value' prefix also
    # searching for everything having a numeric value
    all_margin_values: PatternSearchResults = section.find_sentences_by_pattern_prefix(org_level, self.pattern_factory,
                                                                                       'sum__')

    # s_lawsuits: PatternSearchResults = section.find_sentences_by_pattern_prefix(self.pattern_factory,
    #                                                                             f'x_{ContractSubject.Lawsuit}')

    # s_values: PatternSearchResults = substract_search_results(s_values, s_lawsuits)

    charity_constraints = section.find_sentences_by_pattern_prefix(org_level, self.pattern_factory,
                                                                   f'x_{ContractSubject.Charity}')

    self.map_to_subject(all_margin_values)
    self.map_to_subject(charity_constraints)
    # TODO: if a PatternSearchResult in both charity_constraints & s_values,
    # TODO:    this may re-write the found subject type

    constraints_a: List[ConstraintsSearchResult] = self.__extract_constraint_values_from_sr(all_margin_values)
    constraints_b: List[ConstraintsSearchResult] = self.__extract_constraint_values_from_sr(charity_constraints)

    return constraints_a, constraints_b, all_margin_values, charity_constraints  # TODO: hope there is no intersection

  def get_constraints(self):
    warnings.warn(f'CharterParser.get_constraints are deprecated ☠️! \n Use CharterDocument.value_constraints',
                  DeprecationWarning)
    return self.charter.constraints_old

  def get_charity_constraints(self):
    warnings.warn(
      f'CharterParser.get_charity_constraints are deprecated ☠️! \n Use CharterDocument.charity_constraints',
      DeprecationWarning)
    return self.charter._charity_constraints_old

  def get_org(self):
    warnings.warn(
      f'CharterParser.org are deprecated ☠️! \n Use CharterDocument.org',
      DeprecationWarning)
    return self.charter.org

  def get_charter(self) -> CharterDocument:
    return self.doc

  charity_constraints = property(get_charity_constraints)
  constraints = property(get_constraints)
  charter = property(get_charter)
  org = property(get_org)

  def map_to_subject(self, psearch_results: List[PatternSearchResult]):
    from ml_tools import estimate_confidence

    for psearch_result in psearch_results:
      _max_subj = ContractSubject.Other
      _max_conf = 0.001

      for subj in known_subjects:
        pattern_prefix = f'x_{subj}'

        v = psearch_result.get_attention(AV_PREFIX + pattern_prefix)
        confidence, _, _, _ = estimate_confidence(v)

        if confidence > _max_conf:
          _max_conf = confidence
          _max_subj = subj

      psearch_result.subject_mapping = {
        'confidence': _max_conf,
        'subj': _max_subj
      }

  def __extract_constraint_values_from_sr(self, sentenses_i: PatternSearchResults) -> List[ConstraintsSearchResult]:
    warnings.warn("use __extract_constraint_values_from_sr_2", DeprecationWarning)
    """
    :type sentenses_i: PatternSearchResults
    """
    if not sentenses_i:
      return []

    sentences = []
    for pattern_sr in sentenses_i:
      constraints: List[ProbableValue] = extract_all_contraints_from_sr(pattern_sr, pattern_sr.get_attention())

      # todo: ConstraintsSearchResult is deprecated
      csr = ConstraintsSearchResult()
      csr.subdoc = pattern_sr
      csr.constraints = constraints

      pattern_sr.constraints = constraints
      sentences.append(csr)

    return sentences

  def _do_nothing(self, head, a, b):
    pass  #

  """ 📃️ 🐌 == find_charter_sections_starts ====================================================== """

  # =======================

  def detect_ners(self, section):
    """
    :param section:
    :return:
    """
    assert section is not None

    org_by_type_dict, org_type = self._detect_org_type_and_name(section)
    org_type_tag = org_by_type_dict[org_type]
    start = org_type_tag.span[0]
    start = start + len(self.pattern_factory.patterns_dict[org_type].embeddings)  # typically +1 or +2

    end = 1 + find_ner_end(section.tokens, start)

    orgname_sub_section: LegalDocument = section.subdoc(start, end)
    org_name = orgname_sub_section.tokens_map.text

    # TODO: use same format that is used in agents_info
    rez = {
      'type': org_type,
      'name': org_name,
      'type_name': org_types[org_type],
      'tokens': section.tokens_cc,
      'attention_vector': section.distances_per_pattern_dict[org_type],
    }

    # org_type_span=section.start+
    # org_type_tag = SemanticTag('org_type', org_type, org_type_span)

    org_name_span = section.start + orgname_sub_section.start, section.start + orgname_sub_section.end
    org_name_tag = SemanticTag('org_name', org_name, org_name_span)
    org_type_tag.offset(section.start)

    return rez, org_type_tag, org_name_tag

  def _detect_org_type_and_name(self, section: LegalDocument):

    factory = self.pattern_factory
    vectors = section.distances_per_pattern_dict  # shortcut

    section.calculate_distances_per_pattern(factory, pattern_prefix='org_', merge=True)
    section.calculate_distances_per_pattern(factory, pattern_prefix='ner_org', merge=True)
    section.calculate_distances_per_pattern(factory, pattern_prefix='nerneg_', merge=True)

    vectors['s_attention_vector_neg'] = factory._build_org_type_attention_vector(section)

    org_by_type = {}
    best_org_type = None
    _max = 0
    for org_type in org_types.keys():

      vector = vectors[org_type] * vectors['s_attention_vector_neg']
      if self.verbosity_level > 2:
        print('_detect_org_type_and_name, org_type=', org_type, vectors[org_type][0:10])

      idx = np.argmax(vector)
      val = vectors[org_type][idx]
      if val > _max:
        _max = val
        best_org_type = org_type

      type_name = org_types[org_type]
      tag = SemanticTag('org_type', org_type, (idx, idx + len(type_name.split(' '))))
      tag.confidence = val
      tag.display_value = type_name

      org_by_type[org_type] = tag

    if self.verbosity_level > 2:
      print('_detect_org_type_and_name', org_by_type)

    return org_by_type, best_org_type

    # ==============
    # VIOLATIONS

  def find_ranges_by_group(self, charter_constraints, m_convert, verbose=False):
    return self.violations_finder.find_ranges_by_group(charter_constraints, m_convert, verbose)
    # ranges_by_group = {}
    # for head_group in charter_constraints:
    #   #     print('-' * 20)
    #   group_c = charter_constraints[head_group]
    #   data = self._combine_constraints_in_group(group_c, m_convert, verbose)
    #   ranges_by_group[head_group] = data
    # return ranges_by_group


def put_if_better(destination: dict, key, x, is_better: staticmethod):
  if key in destination:
    if is_better(x, destination[key]):
      destination[key] = x
  else:
    destination[key] = x


def split_by_number_2(tokens: List[str], attention: FixedVector, threshold) -> (
        List[List[str]], List[int], List[slice]):
  indexes = []
  last_token_is_number = False
  for i in range(len(tokens)):

    if attention[i] > threshold and len(number_re.findall(tokens[i])) > 0:
      if not last_token_is_number:
        indexes.append(i)
      last_token_is_number = True
    else:
      last_token_is_number = False

  text_fragments = []
  ranges: List[slice] = []
  if len(indexes) > 0:
    for i in range(1, len(indexes)):
      _slice = slice(indexes[i - 1], indexes[i])
      text_fragments.append(tokens[_slice])
      ranges.append(_slice)

    text_fragments.append(tokens[indexes[-1]:])
    ranges.append(slice(indexes[-1], len(tokens)))
  return text_fragments, indexes, ranges


def extract_sum_and_sign_2(subdoc, region: slice) -> ValueConstraint:
  warnings.warn("deprecated", DeprecationWarning)
  # TODO: rename

  _slice = slice(region.start - VALUE_SIGN_MIN_TOKENS, region.stop)
  subtokens = subdoc.tokens_cc[_slice]
  _prefix_tokens = subtokens[0:VALUE_SIGN_MIN_TOKENS + 1]
  _prefix = untokenize(_prefix_tokens)
  _sign = detect_sign(_prefix)
  # ======================================
  _sum = extract_sum_from_tokens_2(subtokens)
  # ======================================

  currency = "UNDEF"
  value = np.nan
  if _sum is not None:
    currency = _sum[1]
    if _sum[1] in currencly_map:
      currency = currencly_map[_sum[1]]
    value = _sum[0]

  vc = ValueConstraint(value, currency, _sign, TokensWithAttention([], []))

  return vc


def extract_all_contraints_from_sr(search_result: PatternSearchResult, attention_vector: List[float]) -> List[
  ProbableValue]:
  warnings.warn("use extract_all_contraints_from_sr_2", DeprecationWarning)

  def __tokens_before_index(string, index):
    warnings.warn("deprecated", DeprecationWarning)
    return len(string[:index].split(' '))

  sentence = ' '.join(search_result.tokens)
  all_values = [slice(m.start(0), m.end(0)) for m in re.finditer(complete_re, sentence)]
  constraints: List[ProbableValue] = []

  for a in all_values:
    # print(tokens_before_index(sentence, a.start), 'from', sentence[a])
    token_index_s = __tokens_before_index(sentence, a.start) - 1
    token_index_e = __tokens_before_index(sentence, a.stop)

    region = slice(token_index_s, token_index_e)

    vc: ValueConstraint = extract_sum_and_sign_3(search_result, region)
    _e = _expand_slice(region, 10)
    vc.context = TokensWithAttention(search_result.tokens[_e], attention_vector[_e])
    confidence = attention_vector[region.start]
    pv = ProbableValue(vc, confidence)

    constraints.append(pv)

  return constraints


def extract_sum_and_sign_3(sr: PatternMatch, region: slice) -> ValueConstraint:
  warnings.warn("use extract_sum_sign_currency", DeprecationWarning)

  _slice = slice(region.start - VALUE_SIGN_MIN_TOKENS, region.stop)
  subtokens = sr.tokens[_slice]
  _prefix_tokens = subtokens[0:VALUE_SIGN_MIN_TOKENS + 1]
  _prefix = untokenize(_prefix_tokens)
  _sign = detect_sign(_prefix)
  # ======================================
  _sum = extract_sum_from_tokens_2(subtokens)
  # ======================================

  currency = "UNDEF"
  value = np.nan
  if _sum is not None:
    currency = _sum[1]
    if _sum[1] in currencly_map:
      currency = currencly_map[_sum[1]]
    value = _sum[0]

  vc = ValueConstraint(value, currency, _sign, TokensWithAttention([], []))

  return vc


def extract_sum_from_tokens_2(sentence_tokens: Tokens):
  warnings.warn("method relies on untokenize, not good", DeprecationWarning)
  f, __ = extract_sum_from_tokens(sentence_tokens)
  return f


def extract_sum_from_tokens(sentence_tokens: Tokens):
  warnings.warn("method relies on untokenize, not good", DeprecationWarning)
  _sentence = untokenize(sentence_tokens).lower().strip()
  f = extract_sum(_sentence)
  return f, _sentence
