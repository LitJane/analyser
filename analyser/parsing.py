import logging
import time
from functools import wraps
from typing import List

import numpy as np

from analyser.contract_agents import find_closest_org_name
from analyser.contract_patterns import ContractPatternFactory
from analyser.documents import TextMap
from analyser.hyperparams import HyperParameters
from analyser.legal_docs import LegalDocument, ContractValue
from analyser.ml_tools import estimate_confidence_by_mean_top_non_zeros, FixedVector, smooth_safe, relu, SemanticTag
from analyser.schemas import ContractPrice
from analyser.structures import ContractTags
from analyser.transaction_values import complete_re as transaction_values_re, VALUE_SIGN_MIN_TOKENS, ValueSpansFinder, \
  _re_greather_then_1, _re_less_then, _re_greather_then
from gpn.gpn import subsidiaries
from tf_support.embedder_elmo import ElmoEmbedder

PROF_DATA = {}

logger = logging.getLogger('analyser')


class ParsingSimpleContext:
  def __init__(self):

    # ---
    self.verbosity_level = 2
    self.__step = 0

    self.warnings = []

  def _reset_context(self):
    self.warnings = []
    self.__step = 0

  def _logstep(self, name: str) -> None:
    s = self.__step
    logger.info(f'{s}.\t❤️ ACCOMPLISHED:\t {name}')
    self.__step += 1

  def warning(self, text):
    t_ = '\t - ⚠️ WARNING: - ' + text
    self.warnings.append(t_)
    print(t_)

  def get_warings(self):
    return '\n'.join(self.warnings)

  def log_warnings(self):

    if len(self.warnings) > 0:
      logger.warning("Recent analyser warnings:")

      for w in self.warnings:
        logger.warning(w)


class AuditContext:

  def __init__(self, audit_subsidiary_name=None):
    self.audit_subsidiary_name: str = audit_subsidiary_name
    self.fixed_audit_subsidiary_name = '__unknown___'

    known_org_name, _ = find_closest_org_name(subsidiaries, audit_subsidiary_name,
                                                            HyperParameters.subsidiary_name_match_min_jaro_similarity)
    if known_org_name is not None:
      self.fixed_audit_subsidiary_name = known_org_name['_id']

  def is_same_org(self, name: str) -> bool:
    return name in [self.audit_subsidiary_name, self.fixed_audit_subsidiary_name]


class ParsingContext(ParsingSimpleContext):
  def __init__(self, embedder=None, sentence_embedder=None):
    ParsingSimpleContext.__init__(self)
    self._embedder = embedder
    self._sentence_embedder = sentence_embedder

  def get_sentence_embedder(self):
    if self._sentence_embedder is None:
      self._sentence_embedder = ElmoEmbedder.get_instance('default')
    return self._sentence_embedder

  def get_embedder(self):
    if self._embedder is None:
      self._embedder = ElmoEmbedder.get_instance()
    return self._embedder

  def find_org_date_number(self, doc: LegalDocument, ctx: AuditContext) -> LegalDocument:
    """
    phase I, before embedding TF, GPU, and things
    searching for attributes required for filtering
    :param charter:
    :return:
    """
    raise NotImplementedError()

  def find_attributes(self, document: LegalDocument, ctx: AuditContext):
    raise NotImplementedError()

  def validate(self, document: LegalDocument, ctx: AuditContext):
    pass


def profile(fn):
  @wraps(fn)
  @wraps(fn)
  def with_profiling(*args, **kwargs):
    start_time = time.time()

    ret = fn(*args, **kwargs)

    elapsed_time = time.time() - start_time

    if fn.__name__ not in PROF_DATA:
      PROF_DATA[fn.__name__] = [0, []]
    PROF_DATA[fn.__name__][0] += 1
    PROF_DATA[fn.__name__][1].append(elapsed_time)

    return ret

  return with_profiling


def print_prof_data():
  for fname, data in PROF_DATA.items():
    max_time = max(data[1])
    avg_time = sum(data[1]) / len(data[1])
    print("Function {} called {} times. ".format(fname, data[0]))
    print('Execution time max: {:.4f}, average: {:.4f}'.format(max_time, avg_time))


def clear_prof_data():
  global PROF_DATA
  PROF_DATA = {}


head_types_dict = {'head.directors': 'Совет директоров',
                   'head.all': 'Общее собрание участников/акционеров',
                   'head.gen': 'Генеральный директор',
                   #                      'shareholders':'Общее собрание акционеров',
                   'head.pravlenie': 'Правление общества',
                   'head.unknown': '*Неизвестный орган управления*'}
head_types = ['head.directors', 'head.all', 'head.gen', 'head.pravlenie']


def find_value_sign_currency(value_section_subdoc: LegalDocument,
                             factory: ContractPatternFactory = None) -> List[ContractPrice]:
  if factory is not None:
    value_section_subdoc.calculate_distances_per_pattern(factory)
    vectors = factory.make_contract_value_attention_vectors(value_section_subdoc)
    # merge dictionaries of attention vectors
    value_section_subdoc.distances_per_pattern_dict = {**value_section_subdoc.distances_per_pattern_dict, **vectors}

    attention_vector_tuned = value_section_subdoc.distances_per_pattern_dict['value_attention_vector_tuned']
  else:
    # HATI-HATI: this case is for Unit Testing only
    attention_vector_tuned = None

  return find_value_sign_currency_attention(value_section_subdoc, attention_vector_tuned, absolute_spans=True)


def find_value_sign_currency_attention(value_section_subdoc: LegalDocument,
                                       attention_vector_tuned: FixedVector or None,
                                       parent_tag=None,
                                       absolute_spans=False) -> List[ContractPrice]:
  spans = [m for m in value_section_subdoc.tokens_map.finditer(transaction_values_re)]
  values_list: [ContractValue] = []

  for span in spans:
    value_sign_currency: ContractValue = extract_sum_sign_currency(value_section_subdoc, span)
    # TODO: replace with ContractPrice type
    if value_sign_currency is not None:

      # Estimating confidence by looking at attention vector
      if attention_vector_tuned is not None:

        for t in value_sign_currency.as_list():
          t.confidence *= (HyperParameters.confidence_epsilon + estimate_confidence_by_mean_top_non_zeros(
            attention_vector_tuned[t.slice]))
      # ---end if

      value_sign_currency.parent.set_parent_tag(parent_tag)
      value_sign_currency.parent.span = value_sign_currency.span()  ##fix span
      values_list.append(value_sign_currency)

  # offsetting
  if absolute_spans:  # TODO: do not offset here!!!!
    for value in values_list:
      value += value_section_subdoc.start

  return [f.as_ContractPrice() for f in values_list]


def _find_most_relevant_paragraph(section: LegalDocument,
                                  attention_vector: FixedVector,
                                  min_len: int,
                                  return_delimiters=True):
  _blur = HyperParameters.subject_paragraph_attention_blur
  _padding = _blur * 2 + 1

  paragraph_attention_vector = smooth_safe(np.pad(attention_vector, _padding, mode='constant'), _blur)[
                               _padding:-_padding]

  top_index = int(np.argmax(paragraph_attention_vector))
  span = section.sentence_at_index(top_index)
  if min_len is not None and span[1] - span[0] < min_len:
    next_span = section.sentence_at_index(span[1] + 1, return_delimiters)
    span = (span[0], next_span[1])

  confidence_region = attention_vector[span[0]:span[1]]
  confidence = estimate_confidence_by_mean_top_non_zeros(confidence_region)
  return span, confidence, paragraph_attention_vector


def find_most_relevant_paragraphs(section: TextMap,
                                  attention_vector: FixedVector,
                                  min_len: int = 20,
                                  return_delimiters=True, threshold=0.45):
  _blur = int(HyperParameters.subject_paragraph_attention_blur)
  _padding = int(_blur * 2 + 1)

  paragraph_attention_vector = smooth_safe(np.pad(attention_vector, _padding, mode='constant'), _blur)[
                               _padding:-_padding]

  paragraph_attention_vector = relu(paragraph_attention_vector, threshold)

  top_indices = [i for i, v in enumerate(paragraph_attention_vector) if v > 0.00001]
  spans = []
  for i in top_indices:
    span = section.sentence_at_index(i, return_delimiters)
    if min_len is not None and span[1] - span[0] < min_len:
      if span not in spans:
        spans.append(span)

  return spans, paragraph_attention_vector


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