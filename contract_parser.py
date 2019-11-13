from contract_agents import find_org_names
from contract_patterns import ContractPatternFactory
from legal_docs import LegalDocument, extract_sum_sign_currency, ContractValue
from ml_tools import *

from parsing import ParsingConfig, ParsingContext
from patterns import AV_SOFT, AV_PREFIX

from sections_finder import FocusingSectionsFinder
from structures import ContractSubject

default_contract_parsing_config: ParsingConfig = ParsingConfig()
contract_subjects = [ContractSubject.RealEstate, ContractSubject.Charity, ContractSubject.Deal]

from transaction_values import complete_re as transaction_values_re

from hyperparams import HyperParameters


class ContractDocument3(LegalDocument):
  '''

  '''

  # TODO: rename it

  def __init__(self, original_text):
    LegalDocument.__init__(self, original_text)

    self.subjects = None
    self.contract_values: List[ContractValue] = []

    self.agents_tags = None

  def get_tags(self) -> [SemanticTag]:
    tags = []
    tags += self.agents_tags

    if self.subjects:
      tags.append(self.subjects)

    if self.contract_values:
      for contract_value in self.contract_values:
        tags += contract_value.as_list()

    # TODO: filter tags if _t.isNotEmpty():
    return tags


ContractDocument = ContractDocument3  # Alias!


def filter_nans(vcs: List[ProbableValue]) -> List[ProbableValue]:
  warnings.warn("use numpy built-in functions", DeprecationWarning)
  r: List[ProbableValue] = []
  for vc in vcs:
    if vc.value is not None and not np.isnan(vc.value.value):
      r.append(vc)
  return r


class ContractAnlysingContext(ParsingContext):

  def __init__(self, embedder, renderer=None, pattern_factory=None):
    ParsingContext.__init__(self, embedder)

    if not pattern_factory:
      self.pattern_factory = ContractPatternFactory(embedder)
    else:
      self.pattern_factory = pattern_factory

    self.contract = None
    # self.contract_values = None

    self.config = default_contract_parsing_config

    self.sections_finder = FocusingSectionsFinder(self)

  def analyze_contract(self, contract_text):
    warnings.warn("use analyze_contract_doc", DeprecationWarning)

    self._reset_context()
    # create DOC
    self.contract = ContractDocument(contract_text)
    self.contract.parse()

    self._logstep("parsing document 👞 and detecting document high-level structure")
    self.contract.embedd_tokens(self.pattern_factory.embedder)

    return self.analyze_contract_doc(self.contract, reset_ctx=False)

  def analyze_contract_doc(self, contract: ContractDocument, reset_ctx=True):
    # assert contract.embeddings is not None
    # #TODO: this analyser should care about embedding, because it decides wheater it needs (NN) embeddings or not
    """
    MAIN METHOD 2

    :param contract:
    :return:
    
    """
    if reset_ctx:
      self._reset_context()

    self.contract = contract

    # ------ lazy embedding
    if self.contract.embeddings is None:
      self.contract.embedd_tokens(self.pattern_factory.embedder)

    # ------ agents
    contract.agents_tags = find_org_names(contract)
    self._logstep("parsing document 👞 and detecting document high-level structure")

    # ------ structure
    self.sections_finder.find_sections(self.contract, self.pattern_factory, self.pattern_factory.headlines,
                                       headline_patterns_prefix='headline.')

    # -------------------------------values
    self.contract.contract_values = self.find_contract_value_NEW(self.contract)
    self._logstep("finding contract values")

    # -------------------------------subject
    self.contract.subjects = self.find_contract_subject_region(self.contract)
    self._logstep("detecting contract subject")
    # --------------------------------------

    self.log_warnings()

    return self.contract, self.contract.contract_values

  def _reset_context(self):
    super(ContractAnlysingContext, self)._reset_context()

    if self.contract is not None:
      del self.contract
      self.contract = None

  def get_contract_values(self):
    return self.contract.contract_values

  contract_values = property(get_contract_values)

  def select_most_confident_if_almost_equal(self, a: ProbableValue, alternative: ProbableValue, m_convert,
                                            equality_range=0.0):

    if abs(m_convert(a.value).value - m_convert(alternative.value).value) < equality_range:
      if a.confidence > alternative.confidence:
        return a
      else:
        return alternative
    return a

  def find_contract_best_value(self, m_convert):
    best_value: ProbableValue = max(self.contract_values,
                                    key=lambda item: m_convert(item.value).value)

    most_confident_value = max(self.contract_values, key=lambda item: item.confidence)
    best_value = self.select_most_confident_if_almost_equal(best_value, most_confident_value, m_convert,
                                                            equality_range=20)

    return best_value

  def __sub_attention_names(self, subj: ContractSubject):
    a = f'x_{subj}'
    b = AV_PREFIX + f'x_{subj}'
    c = AV_SOFT + a
    return a, b, c

  def make_subject_attention_vector_3(self, section, subject_kind: ContractSubject, addon=None) -> FixedVector:

    pattern_prefix, attention_vector_name, attention_vector_name_soft = self.__sub_attention_names(subject_kind)

    vectors = filter_values_by_key_prefix(section.distances_per_pattern_dict, pattern_prefix)
    if addon is not None:
      vectors = list(vectors)
      vectors.append(addon)
    x = max_exclusive_pattern(vectors)
    assert x is not None, f'no patterns for {subject_kind}'

    section.distances_per_pattern_dict[attention_vector_name_soft] = x
    section.distances_per_pattern_dict[attention_vector_name] = x

    #   x = x-np.mean(x)
    x = relu(x, 0.6)

    return x

  def find_contract_subject_region(self, doc) -> SemanticTag:
    if 'subj' in doc.sections:
      subj_section = doc.sections['subj']
      subject_subdoc = subj_section.body
      denominator = 1
    else:
      self.warning('раздел о предмете договора не найден, ищем предмет договора в первых 1500 словах')
      subject_subdoc = doc.subdoc_slice(slice(0, 1500))
      denominator = 0.7

    return self.find_contract_subject_regions(subject_subdoc, denominator=denominator)

  def find_contract_subject_regions(self, section: LegalDocument, denominator: float = 1.0) -> SemanticTag:
    # TODO: build trainset on contracts, train simple model for detectin start and end of contract subject region
    # TODO: const(loss) function should measure distance from actual span to expected span

    section.calculate_distances_per_pattern(self.pattern_factory, merge=True, pattern_prefix='x_ContractSubject')
    section.calculate_distances_per_pattern(self.pattern_factory, merge=True, pattern_prefix='headline.subj')

    all_subjects_vectors = filter_values_by_key_prefix(section.distances_per_pattern_dict, 'headline.subj')
    subject_headline_attention: FixedVector = rectifyed_sum(all_subjects_vectors) / 2

    max_confidence = 0
    max_subject_kind = None
    max_paragraph_span = None

    for subject_kind in contract_subjects:  # like ContractSubject.RealEstate ..
      subject_attention_vector: FixedVector = self.make_subject_attention_vector_3(section, subject_kind,
                                                                                   subject_headline_attention)

      paragraph_span, confidence, paragraph_attention_vector = _find_most_relevant_paragraph(section,
                                                                                             subject_attention_vector,
                                                                                             min_len=20,
                                                                                             return_delimiters=False)

      if self.verbosity_level > 2:
        print(f'--------------------confidence {subject_kind}=', confidence)
      if confidence > max_confidence:
        max_confidence = confidence
        max_subject_kind = subject_kind
        max_paragraph_span = paragraph_span

    if max_subject_kind:
      subject_tag = SemanticTag('subject', max_subject_kind.name, max_paragraph_span)
      subject_tag.confidence = max_confidence * denominator
      subject_tag.offset(section.start)

      return subject_tag

  def find_contract_value_NEW(self, contract: ContractDocument) -> List[ContractValue]:
    # preconditions
    assert contract.sections is not None, 'find sections first'

    search_sections_order = [
      ['cvalue', 1], ['pricecond', 0.75], ['subj', 0.75], [None, 0.5]  # todo: check 'price', not 'price.'
    ]

    for section, confidence_k in search_sections_order:
      if section in contract.sections or section is None:

        if section in contract.sections:
          value_section = contract.sections[section].body
          _section_name = contract.sections[section].subdoc.text.strip()
        else:
          value_section = contract
          _section_name = 'entire contract'

        if self.verbosity_level > 1:
          self._logstep(f'searching for transaction values in section ["{section}"] "{_section_name}"')

        values_list: List[ContractValue] = find_value_sign_currency(value_section, self.pattern_factory)

        if not values_list:
          # search in next section
          self.warning(f'В разделе "{_section_name}" ["{section}"] стоимость сделки не найдена!')

        else:
          # decrease confidence:
          for g in values_list:
            for _r in g.as_list():
              _r.confidence *= confidence_k


          # ------
          # reduce number of found values
          # take only max value and most confident ones (we hope, it is the same finding)

          max_confident_cv = max_confident(values_list)
          max_valued_cv = max_value(values_list)
          if max_confident_cv == max_valued_cv:
            return [max_confident_cv]
          else:
            return [max_valued_cv, max_confident_cv]


      else:
        self.warning(f'Раздел [{section}]  не обнаружен')


def find_value_sign_currency(value_section_subdoc: LegalDocument, factory: ContractPatternFactory = None) -> List[
  ContractValue]:
  if factory is not None:
    value_section_subdoc.calculate_distances_per_pattern(factory)
    vectors = factory.make_contract_value_attention_vectors(value_section_subdoc)
    # merge dictionaries of attention vectors
    value_section_subdoc.distances_per_pattern_dict = {**value_section_subdoc.distances_per_pattern_dict, **vectors}

    attention_vector_tuned = value_section_subdoc.distances_per_pattern_dict['value_attention_vector_tuned']
  else:
    # HATI-HATI: this case is for Unit Testing only
    attention_vector_tuned = None

  return find_value_sign_currency_attention(value_section_subdoc, attention_vector_tuned)


def find_value_sign_currency_attention(value_section_subdoc: LegalDocument, attention_vector_tuned=None) -> List[
  ContractValue]:
  spans = [m for m in value_section_subdoc.tokens_map.finditer(transaction_values_re)]
  values_list = []

  for span in spans:
    value_sign_currency = extract_sum_sign_currency(value_section_subdoc, span)
    if value_sign_currency is not None:

      # Estimating confidence by looking at attention vector
      if attention_vector_tuned is not None:
        for t in value_sign_currency.as_list():
          t.confidence *= (HyperParameters.confidence_epsilon + estimate_confidence_by_mean_top_non_zeros(
            attention_vector_tuned[t.slice]))
          t.offset(value_section_subdoc.start)

      values_list.append(value_sign_currency)

  return values_list


def max_confident(vals: List[ContractValue]) -> ContractValue:
  return max(vals, key=lambda a: a.integral_sorting_confidence())


def max_value(vals: List[ContractValue]) -> ContractValue:
  return max(vals, key=lambda a: a.value.value)


def _find_most_relevant_paragraph(section: LegalDocument, subject_attention_vector: FixedVector, min_len: int,
                                  return_delimiters=True):
  # paragraph_attention_vector = smooth(attention_vector, 6)

  _blur = HyperParameters.subject_paragraph_attention_blur
  _padding = _blur * 2 + 1

  paragraph_attention_vector = smooth_safe(np.pad(subject_attention_vector, _padding, mode='constant'), _blur)[
                               _padding:-_padding]

  top_index = int(np.argmax(paragraph_attention_vector))
  span = section.tokens_map.sentence_at_index(top_index)
  if min_len is not None and span[1] - span[0] < min_len:
    next_span = section.tokens_map.sentence_at_index(span[1] + 1, return_delimiters)
    span = (span[0], next_span[1])

  # confidence = paragraph_attention_vector[top_index]
  confidence_region = subject_attention_vector[span[0]:span[1]]

  # print(confidence_region)
  confidence = estimate_confidence_by_mean_top_non_zeros(confidence_region)
  return span, confidence, paragraph_attention_vector


def find_all_value_sign_currency(doc: LegalDocument) -> List[ContractValue]:
  warnings.warn("use find_value_sign_currency ", DeprecationWarning)
  """
  TODO: rename
  :param doc: LegalDocument
  :param attention_vector: List[float]
  :return: List[ProbableValue]
  """
  spans = [m for m in doc.tokens_map.finditer(transaction_values_re)]
  return [extract_sum_sign_currency(doc, span) for span in spans]


extract_all_contraints_from_sr_2 = find_all_value_sign_currency  # alias for compatibility, todo: remove it
