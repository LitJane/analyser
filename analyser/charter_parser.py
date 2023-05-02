# origin: charter_parser.py
import os
import warnings

import pandas as pd
# from overrides import overrides
from pandas import DataFrame

from analyser.attributes import to_json
from analyser.contract_agents import find_org_names, ContractAgent, find_org_names_raw
from analyser.doc_dates import find_document_date
from analyser.embedding_tools import AbstractEmbedder
from analyser.hyperparams import HyperParameters, models_path
from analyser.legal_docs import LegalDocumentExt, remap_attention_vector, embedd_sentences, LegalDocument, \
  ParserWarnings
from analyser.ml_tools import SemanticTag, calc_distances_per_pattern, merge_colliding_spans, Spans, \
  FixedVector, span_to_slice, estimate_confidence_by_mean_top_non_zeros, calc_distances_per_pattern_dict, \
  max_exclusive_pattern_by_prefix, relu, attribute_patternmatch_to_index, SemanticTagBase
from analyser.parsing import ParsingContext, AuditContext, find_value_sign_currency_attention, \
  _find_most_relevant_paragraph
from analyser.patterns import build_sentence_patterns, PATTERN_DELIMITER
from analyser.schemas import CharterSchema, CharterStructuralLevel, Competence, ContractPrice
from analyser.structures import OrgStructuralLevel, ContractSubject

WARN = '\033[1;31m'

competence_headline_pattern_prefix = 'headline'

number_key = SemanticTag.number_key


class CharterDocument(LegalDocumentExt):

  def __init__(self, doc: LegalDocument = None):
    super().__init__(doc)
    if doc is not None:
      self.__dict__.update(doc.__dict__)
    self.attributes_tree = CharterSchema()

  def reset_attributes(self):
    """
    reset for preventing doubling tags
    :return:
    """
    self.attributes_tree = CharterSchema()

  def get_number(self) -> SemanticTagBase:
    return self.attributes_tree.number

  def set_number(self, number):
    self.attributes_tree.number = number

  def get_date(self) -> SemanticTagBase:
    warnings.warn("use attributes_tree.date", DeprecationWarning)
    return self.attributes_tree.date

  def set_date(self, date):
    warnings.warn("use attributes_tree.date", DeprecationWarning)
    self.attributes_tree.date = date

  date = property(get_date, set_date)
  number = property(get_number, set_number)

  #   @overrides
  def to_json_obj(self) -> dict:
    j: dict = super().to_json_obj()
    _attributes_tree_dict, _ = to_json(self.attributes_tree)
    j['attributes_tree']['charter'] = _attributes_tree_dict
    return j


def _make_org_level_patterns() -> pd.DataFrame:
  p = os.path.join(models_path, 'charter_org_level_patterns.json')
  comp_str_pat = pd.read_json(p, orient='index')
  return comp_str_pat.astype('str')


class CharterParser(ParsingContext):
  strs_subjects_patterns = {

    ContractSubject.Deal: [
      'принятие решений о совершении сделок'
    ],

    ContractSubject.BigDeal: [
      'совершение крупных сделок',
      'согласие на совершение или одобрение крупных сделок'
    ],

    ContractSubject.Charity: [
      "оплата (встречное предоставление) в неденежной форме",
      "пожертвования на политические или благотворительные цели",
      "предоставление безвозмездной финансовой помощи",
      "сделки дарения",
      'безвозмездное отчуждение имущества',
      "договоры спонсорского и благотворительного характера",
      "передача в безвозмездное пользование",
      "мена, дарение, безвозмездное отчуждение",
      'внесение вкладов или пожертвований на политические или благотворительные цели'
    ],

    ContractSubject.Lawsuit: [
      'урегулирование любых судебных споров и разбирательств',
      'заключение мирового соглашения по судебному делу с ценой иска '
    ],

    ContractSubject.RealEstate: [
      'сделки с имуществом Общества',
      'стоимость отчуждаемого имущества',
      'сделок ( в том числе нескольких взаимосвязанных сделок ) с имуществом Общества'
    ],

    ContractSubject.Insurance: [
      'заключение договоров страхования',
      'возобновления договоров страхования',
      'совершение сделок страхования'
    ],

    ContractSubject.Service: [
      'оказания консультационных услуг',
      'заключение агентского договора',
      'оказание обществу информационных юридических услуг'
    ],

    # CharterSubject.Other: [
    #   'решения о взыскании с Генерального директора убытков',
    #   'заключение договоров об отступном, новации или прощении долга, договоров об уступке права требования и переводе долга',
    #   'нецелевое расходование Обществом денежных средств'
    # ],

    ContractSubject.Loans: [
      'получение или предоставление займов, кредитов (в том числе вексельных)',
      'предоставление гарантий и поручительств по обязательствам',
      'предоставление займа или получения заимствования, кредита, финансирования, выплаты или отсрочки по займу, кредиту, финансированию или задолженности',
      'предоставление обеспечений исполнения обязательств',
      'получение банковских гарантий'
      # 'о выдаче или получении Обществом векселей, производстве по ним передаточных надписей, авалей, платежей',
    ],

    ContractSubject.Renting: [
      'получение в аренду или субаренду недвижимого имущества',
      'о совершении сделок, связанных с получением в аренду недвижимоcти'
    ],

    ContractSubject.RentingOut: [
      'передача в аренду или субаренду недвижимого имущества',
      'о совершении сделок, связанных с передачей в аренду недвижимоcти'

    ]

  }

  def __init__(self, embedder: AbstractEmbedder = None, sentence_embedder: AbstractEmbedder = None):
    ParsingContext.__init__(self, embedder, sentence_embedder)

    self.patterns_dict: DataFrame = _make_org_level_patterns()

    self._patterns_named_embeddings: DataFrame or None = None
    self._subj_patterns_embeddings = None

  def get_patterns_named_embeddings(self):
    if self._patterns_named_embeddings is None:
      __patterns_embeddings = self.get_sentence_embedder().embedd_strings(self.patterns_dict.values[0])
      self._patterns_named_embeddings = pd.DataFrame(__patterns_embeddings.T, columns=self.patterns_dict.columns)

    return self._patterns_named_embeddings

  def get_subj_patterns_embeddings(self):

    if self._subj_patterns_embeddings is None:
      self._subj_patterns_embeddings = embedd_charter_subject_patterns(CharterParser.strs_subjects_patterns,
                                                                       self.get_embedder())

    return self._subj_patterns_embeddings

  def _embedd(self, charter: CharterDocument):

    ### ⚙️🔮 SENTENCES embedding

    charter.sentences_embeddings = embedd_sentences(charter.sentence_map, self.get_sentence_embedder(), log_addon=str(charter.get_id()))
    charter.distances_per_sentence_pattern_dict = calc_distances_per_pattern(charter.sentences_embeddings,
                                                                             self.get_patterns_named_embeddings())

  def find_org_date_number(self, doc: CharterDocument, ctx: AuditContext) -> LegalDocument:
    """
    phase 1, before embedding
    searching for attributes required for filtering
    :param charter:
    :return:
    """
    # charter.sentence_map = tokenize_doc_into_sentences_map(charter, HyperParameters.charter_sentence_max_len)

    # doc.org_tags = find_charter_org(doc)
    doc.attributes_tree.org = find_charter_org_obj(doc)
    if doc.attributes_tree.org:
      doc.attributes_tree.org.alias = None
    doc.attributes_tree.date = find_document_date(doc)

    return doc

  def find_attributes(self, _charter: CharterDocument, ctx: AuditContext) -> CharterDocument:
    # reset for preventing tags doubling
    _charter.reset_attributes()

    self.find_org_date_number(_charter, ctx)

    if _charter.sentences_embeddings is None:
      # lazy embedding
      self._embedd(_charter)

    # --------------
    # (('Pattern name', 16), 0.8978644013404846),
    patterns_by_headers = map_headlines_to_patterns(_charter,
                                                    self.get_patterns_named_embeddings(), self.get_sentence_embedder())

    _parent_org_level_tag_keys = []
    for p_mapping in patterns_by_headers:
      # for each 'competence' article
      _pattern_name = p_mapping[0][0]
      _paragraph_id = p_mapping[0][1]

      paragraph_body: SemanticTag = _charter.paragraphs[_paragraph_id].body
      confidence = p_mapping[1]
      _org_level_name = _pattern_name.split(PATTERN_DELIMITER)[-1]
      org_level: OrgStructuralLevel = OrgStructuralLevel[_org_level_name]
      subdoc = _charter.subdoc_slice(paragraph_body.as_slice())
      # --
      structurallevel = CharterStructuralLevel()
      structurallevel.value = org_level
      structurallevel.set_confidence(confidence)
      structurallevel.set_span(paragraph_body.span)
      _charter.attributes_tree.structural_levels.append(structurallevel)

      self.find_attributes_in_sections(subdoc, structurallevel)

    return _charter

  def find_attributes_in_sections(self, subdoc: LegalDocumentExt, structural_level: CharterStructuralLevel):

    # finding Subjects
    _subject_attentions_map = get_charter_subj_attentions(subdoc, self.get_subj_patterns_embeddings())  # dictionary
    subject_spans: Spans = collect_subjects_spans2(subdoc, _subject_attentions_map)

    # finding Values(amounts)
    values: [ContractPrice] = find_value_sign_currency_attention(subdoc, None, absolute_spans=False)
    # self._rename_margin_values_tags(values)
    valued_sentence_spans: Spans = collect_sentences_having_constraint_values(subdoc, values, merge_spans=True)

    _united_spans: Spans = []
    for _s in valued_sentence_spans:
      _united_spans.append(_s)
    for _s in subject_spans:
      _united_spans.append(_s)

    _united_spans = merge_colliding_spans(_united_spans, eps=-1)  # TODO: check this

    self.attribute_spans_to_subjects(_united_spans,
                                     subdoc,
                                     structural_level  # OrgStructuralLevel.BoardOfDirectors
                                     )

    # --------------------
    # offsetting tags to absolute values
    for value in values:
      value += subdoc.start
    # --------------------
    for competence_tag in structural_level.competences:
      competence_tag += subdoc.start

    # nesting values (assigning parents)
    for competence in structural_level.competences:  # contract subjects

      for value in values:
        if competence.contains(value.get_span()):
          competence.constraints.append(value)

  def attribute_spans_to_subjects(self,
                                  unique_sentence_spans: Spans,
                                  subdoc: LegalDocumentExt,
                                  parent_org_level_tag: CharterStructuralLevel):

    subject_attentions_map: dict = get_charter_subj_attentions(subdoc, self.get_subj_patterns_embeddings())
    all_subjects = list(subject_attentions_map.keys())
    parent_org_level_tag.competences = []
    # attribute sentences to subject
    for contract_value_sentence_span in unique_sentence_spans:

      max_confidence = 0
      best_subject = None

      for subj in all_subjects:
        av: FixedVector = subject_attentions_map[subj]

        confidence_region: FixedVector = av[span_to_slice(contract_value_sentence_span)]
        confidence = estimate_confidence_by_mean_top_non_zeros(confidence_region)

        if confidence > max_confidence:
          max_confidence = confidence
          best_subject = subj
      # end for

      if best_subject is not None:
        competence = Competence()
        competence.value = best_subject
        competence.set_span(contract_value_sentence_span)
        competence.set_confidence(max_confidence)

        parent_org_level_tag.competences.append(competence)

        all_subjects.remove(best_subject)  # taken: avoid duplicates


def collect_sentences_having_constraint_values(subdoc: LegalDocumentExt, contract_values: [ContractPrice],
                                               merge_spans=True) -> Spans:
  # collect sentences having constraint values
  unique_sentence_spans: Spans = []
  for contract_value in contract_values:
    parent_span = contract_value.get_span()
    contract_value_sentence_span = subdoc.sentence_at_index(parent_span[0], return_delimiters=False)
    if contract_value_sentence_span not in unique_sentence_spans:
      unique_sentence_spans.append(contract_value_sentence_span)
    contract_value_sentence_span = subdoc.sentence_at_index(parent_span[1], return_delimiters=False)
    if contract_value_sentence_span not in unique_sentence_spans:
      unique_sentence_spans.append(contract_value_sentence_span)
  # --
  # TODO: do not join here, join by subject
  if merge_spans:
    unique_sentence_spans = merge_colliding_spans(unique_sentence_spans, eps=1)
  return unique_sentence_spans


def embedd_charter_subject_patterns(patterns_dict, embedder: AbstractEmbedder):
  emb_subj_patterns = {}
  for subj in patterns_dict.keys():
    strings = patterns_dict[subj]
    prefix = PATTERN_DELIMITER.join(['subject', subj.name])

    emb_subj_patterns[subj] = {
      'patterns': build_sentence_patterns(strings, prefix, subj),
      'embedding': embedder.embedd_strings(strings)
    }

  return emb_subj_patterns


def get_charter_subj_attentions(subdoc: LegalDocumentExt, emb_subj_patterns) -> dict:
  # TODO: use pandas
  _distances_per_subj = {}

  for subj in emb_subj_patterns.keys():
    patterns_distances = calc_distances_per_pattern_dict(subdoc.sentences_embeddings,
                                                         emb_subj_patterns[subj]['patterns'],
                                                         emb_subj_patterns[subj]['embedding'])

    prefix = PATTERN_DELIMITER.join(['subject', subj.name])

    subj_av = relu(max_exclusive_pattern_by_prefix(patterns_distances, prefix), 0.6)  # TODO: use hyper parameter
    subj_av_words = remap_attention_vector(subj_av, subdoc.sentence_map, subdoc.tokens_map)

    _distances_per_subj[subj] = subj_av_words

  return _distances_per_subj


def collect_subjects_spans2(subdoc, subject_attentions_map, min_len=20) -> Spans:
  spans = []
  for subj in subject_attentions_map.keys():

    subject_attention = subject_attentions_map[subj]
    paragraph_span, confidence, _ = _find_most_relevant_paragraph(subdoc,
                                                                  subject_attention,
                                                                  min_len=min_len,
                                                                  return_delimiters=False)
    if confidence > HyperParameters.charter_subject_attention_confidence:
      if paragraph_span not in spans:
        spans.append(paragraph_span)

  unique_sentence_spans = merge_colliding_spans(spans, eps=-1)

  return unique_sentence_spans


def find_charter_org_obj(doc: LegalDocument) -> ContractAgent or None:
  _subdoc = doc[0:HyperParameters.protocol_caption_max_size_words]

  orgs: [ContractAgent] = find_org_names_raw(_subdoc, max_names=1)
  if len(orgs) == 0:
    return None

  return orgs[0]


def find_charter_org(charter: LegalDocument) -> [SemanticTag]:
  """
  TODO: see also find_protocol_org
  :param charter:
  :return:
  """
  ret = []
  x: [SemanticTag] = find_org_names(charter[0:HyperParameters.protocol_caption_max_size_words], max_names=1)
  nm = SemanticTag.find_by_kind(x, 'org-1-name')
  if nm is not None:
    ret.append(nm)
  else:
    charter.warn(ParserWarnings.org_name_not_found)

  tp = SemanticTag.find_by_kind(x, 'org-1-type')
  if tp is not None:
    ret.append(tp)
  else:
    charter.warn(ParserWarnings.org_type_not_found)

  return ret


def map_headlines_to_patterns(doc: LegalDocument,
                              patterns_named_embeddings: DataFrame,
                              elmo_embedder_default: AbstractEmbedder):
  warnings.warn("consider using map_headers, it returns probalility distribution", DeprecationWarning)
  headers: [str] = doc.headers_as_sentences()

  if not headers:
    return []

  headers_embedding = elmo_embedder_default.embedd_strings(headers)

  header_to_pattern_distances = calc_distances_per_pattern(headers_embedding, patterns_named_embeddings)
  return attribute_patternmatch_to_index(header_to_pattern_distances)
