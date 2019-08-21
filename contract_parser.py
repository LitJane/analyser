import warnings
from typing import List

from contract_agents import agent_infos_to_tags, find_org_names_spans
from contract_patterns import ContractPatternFactory
from legal_docs import LegalDocument, HeadlineMeta, extract_sum_sign_currency
from ml_tools import ProbableValue, relu, np, filter_values_by_key_prefix, \
  rectifyed_sum, SemanticTag, FixedVector
from parsing import ParsingConfig, ParsingContext
from patterns import AV_SOFT, AV_PREFIX
from renderer import AbstractRenderer
from sections_finder import SectionsFinder, FocusingSectionsFinder
from structures import ContractSubject
from transaction_values import ValueConstraint

default_contract_parsing_config: ParsingConfig = ParsingConfig()
contract_subjects = [ContractSubject.RealEstate, ContractSubject.Charity, ContractSubject.Deal]

from transaction_values import complete_re as transaction_values_re


class ContractDocument3(LegalDocument):
  '''

  '''

  # TODO: rename it

  def __init__(self, original_text):
    LegalDocument.__init__(self, original_text)
    self.subjects: List[ProbableValue] = [ProbableValue(ContractSubject.Other, 0.0)]
    self.contract_values: [ProbableValue] = []

    self.agents_tags = None

  def get_tags(self) -> [SemanticTag]:
    return self.agents_tags

  def parse(self, txt=None):
    super().parse()
    agent_infos = find_org_names_spans(self.tokens_map_norm)
    self.agents_tags = agent_infos_to_tags(agent_infos)


ContractDocument = ContractDocument3  # Alias!


def estimate_confidence_2(x, head_size: int = 10) -> float:
  """
  taking mean of max 10 values
  """
  return float(np.mean(sorted(x)[-head_size:]))


class ContractAnlysingContext(ParsingContext):

  def __init__(self, embedder, renderer: AbstractRenderer):
    ParsingContext.__init__(self, embedder)
    self.renderer: AbstractRenderer = renderer
    self.pattern_factory = ContractPatternFactory(embedder)

    self.contract = None
    # self.contract_values = None

    self.config = default_contract_parsing_config

    # self.sections_finder: SectionsFinder = DefaultSectionsFinder(self)
    self.sections_finder: SectionsFinder = FocusingSectionsFinder(self)

  def _reset_context(self):
    super(ContractAnlysingContext, self)._reset_context()

    if self.contract is not None:
      del self.contract
      self.contract = None

  def analyze_contract(self, contract_text):
    self._reset_context()
    """
    MAIN METHOD
    
    :param contract_text: 
    :return: 
    """
    doc = ContractDocument(contract_text)
    doc.parse()
    self.contract = doc

    self._logstep("parsing document 👞 and detecting document high-level structure")

    self.contract.embedd_tokens(self.pattern_factory.embedder)
    self.sections_finder.find_sections(doc, self.pattern_factory, self.pattern_factory.headlines,
                                       headline_patterns_prefix='headline.')

    # -------------------------------values
    doc.contract_values = self.fetch_value_from_contract(doc)
    # -------------------------------subject
    doc.subjects = self.recognize_subject(doc)
    self._logstep("fetching transaction values")

    self.renderer.render_values(doc.contract_values)
    self.log_warnings()

    return doc, doc.contract_values

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
    from ml_tools import max_exclusive_pattern
    pattern_prefix, attention_vector_name, attention_vector_name_soft = self.__sub_attention_names(subject_kind)

    vectors = filter_values_by_key_prefix(section.distances_per_pattern_dict, pattern_prefix)
    x = max_exclusive_pattern(vectors)

    section.distances_per_pattern_dict[attention_vector_name_soft] = x
    section.distances_per_pattern_dict[attention_vector_name] = x

    #   x = x-np.mean(x)
    x = relu(x, 0.6)

    return x

  def map_subject_to_type(self, section: LegalDocument, denominator: float = 1.0) -> List[ProbableValue]:
    """
    :param section:
    :param denominator: confidence multiplyer
    :return:
    """
    section.calculate_distances_per_pattern(self.pattern_factory, merge=True, pattern_prefix='x_ContractSubject')
    all_subjects_vectors = filter_values_by_key_prefix(section.distances_per_pattern_dict, 'x_ContractSubject')
    all_subjects_mean: FixedVector = rectifyed_sum(all_subjects_vectors)

    subjects_mapping: List[ProbableValue] = []
    for subject_kind in contract_subjects:  # like ContractSubject.RealEstate ..
      x: FixedVector = self.make_subject_attention_vector_3(section, subject_kind, all_subjects_mean)

      confidence = estimate_confidence_2(x)
      confidence *= denominator
      pv = ProbableValue(subject_kind, confidence)
      subjects_mapping.append(pv)

    return subjects_mapping

  def recognize_subject(self, doc) -> List[ProbableValue]:

    if 'subj' in doc.sections:
      subj_section = doc.sections['subj']
      subj_ = subj_section.body

      return self.map_subject_to_type(subj_)

    else:
      self.warning('раздел о предмете договора не найден')
      # try:
      self.warning('ищем предмет договора в первых 1500 словах')

      return self.map_subject_to_type(doc.subdoc_slice(slice(0, 1500)), denominator=0.7)
      # except:
      #   self.warning('поиск предмета договора полностью провален!')
      #   return [ProbableValue(ContractSubject.Other, 0.0)]

  def fetch_value_from_contract(self, contract: LegalDocument) -> List[ProbableValue]:

    assert contract.sections is not None

    def filter_nans(vcs: List[ProbableValue]) -> List[ProbableValue]:
      warnings.warn("use numpy built-in functions", DeprecationWarning)
      r: List[ProbableValue] = []
      for vc in vcs:
        if vc.value is not None and not np.isnan(vc.value.value):
          r.append(vc)
      return r

    renderer = self.renderer
    price_factory = self.pattern_factory
    sections = contract.sections
    result: List[ValueConstraint] = []

    # TODO iterate over section names
    if 'price.' in sections:  # todo: check 'price', not 'price.'

      value_section_info: HeadlineMeta = sections['price.']
      value_section = value_section_info.body
      section_name = value_section_info.subdoc.text
      result = filter_nans(_try_to_fetch_value_from_section_2(value_section, price_factory))
      if len(result) == 0:
        self.warning(f'В разделе "{section_name}" стоимость сделки не найдена!')

      if self.verbosity_level > 1:
        renderer.render_value_section_details(value_section_info)
        self._logstep(f'searching for transaction values in section  "{section_name}"')
        # ------------
        # value_section.reset_embeddings()  # careful with this. Hope, we will not be required to search here
    else:
      self.warning('Раздел про стоимость сделки не найден!')

    if len(result) == 0:
      if 'subj' in sections:

        # fallback
        value_section_info = sections['subj']
        value_section = value_section_info.body
        section_name = value_section_info.subdoc.text
        print(f'- Ищем стоимость в разделе {section_name}')
        result: List[ProbableValue] = filter_nans(_try_to_fetch_value_from_section_2(value_section, price_factory))

        # decrease confidence:
        for _r in result:
          _r.confidence *= 0.7

        if self.verbosity_level > 0:
          print('alt price section DOC', '-' * 20)
          renderer.render_value_section_details(value_section_info)
          self._logstep(f'searching for transaction values in section  "{section_name}"')

        if len(result) == 0:
          self.warning(f'В разделе "{section_name}" стоимость сделки не найдена!')

    if len(result) == 0:
      if 'pricecond' in sections:

        # fallback
        value_section_info = sections['pricecond']
        value_section = value_section_info.body
        section_name = value_section_info.subdoc.text
        print(f'-WARNING: Ищем стоимость в разделе {section_name}!')
        result: List[ProbableValue] = filter_nans(_try_to_fetch_value_from_section_2(value_section, price_factory))
        if self.verbosity_level > 0:
          print('alt price section DOC', '-' * 20)
          renderer.render_value_section_details(value_section_info)
          self._logstep(f'searching for transaction values in section  "{section_name}"')
        # ------------
        for _r in result:
          _r.confidence *= 0.7
        # value_section.reset_embeddings()  # careful with this. Hope, we will not be required to search here
        if len(result) == 0:
          self.warning(f'В разделе "{section_name}" стоимость сделки не найдена!')

    if len(result) == 0:
      self.warning('Ищем стоимость во всем документе!')

      #     trying to find sum in the entire doc
      value_section = contract
      result: List[ProbableValue] = filter_nans(_try_to_fetch_value_from_section_2(value_section, price_factory))
      if self.verbosity_level > 1:
        print('ENTIRE DOC', '--' * 70)
        self._logstep(f'searching for transaction values in the entire document')
      # ------------
      # decrease confidence:
      for _r in result:
        _r.confidence *= 0.6
      # value_section.reset_embeddings()  # careful with this. Hope, we will not be required to search here

    return result


def _try_to_fetch_value_from_section_2(value_section_subdoc: LegalDocument, factory: ContractPatternFactory) -> List[
  ProbableValue]:
  ''' merge dictionaries of attention vectors '''

  value_section_subdoc.calculate_distances_per_pattern(factory)
  vectors = factory.make_contract_value_attention_vectors(value_section_subdoc)

  value_section_subdoc.distances_per_pattern_dict = {**value_section_subdoc.distances_per_pattern_dict, **vectors}

  v = value_section_subdoc.distances_per_pattern_dict['value_attention_vector_tuned']
  values: List[ProbableValue] = find_all_value_sign_currency(value_section_subdoc)

  return values


def extract_all_contraints_from_sr_2(doc: LegalDocument) -> List:
  """
  TODO: rename
  :param doc: LegalDocument
  :param attention_vector: List[float]
  :return: List[ProbableValue]
  """
  spans = [m for m in doc.tokens_map.finditer(transaction_values_re)]
  return [extract_sum_sign_currency(doc, span) for span in spans]


find_all_value_sign_currency = extract_all_contraints_from_sr_2
