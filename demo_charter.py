# ------------------------------

from demo import match_headline_types, find_sections_by_headlines
from legal_docs import LegalDocument, untokenize, embedd_generic_tokenized_sentences, HeadlineMeta
from legal_docs import extract_all_contraints_from_sentence, \
  embedd_headlines, deprecated, CharterDocument, make_soft_attention_vector, make_constraints_attention_vectors
from ml_tools import *
from patterns import AbstractPatternFactory, FuzzyPattern
from renderer import AbstractRenderer
from text_tools import find_ner_end
from transaction_values import ValueConstraint, extract_sum


class CharterAnlysingContext:
  def __init__(self, embedder, renderer: AbstractRenderer):
    assert embedder is not None
    assert renderer is not None

    self.price_factory = CharterConstraintsPatternFactory(embedder)
    self.hadlines_factory = CharterHeadlinesPatternFactory(embedder)
    self.ner_factory = CharterNerPatternFactory(embedder)
    self.renderer = renderer

    self.org = None
    self.constraints = None
    self.doc = None


  def analyze_charter(self, txt, verbose=False):
    # parse
    _charter_doc = CharterDocument(txt)
    _charter_doc.right_padding = 0
    _charter_doc.parse()

    # 1. find top level structure
    #   headline_indexes = _charter_doc.structure.get_lines_by_level(0)

    headline_indexes = _charter_doc.structure.headline_indexes

    # 2. embedd headlines
    embedded_headlines = embedd_headlines(headline_indexes, _charter_doc, self.hadlines_factory)

    # 3. apply semantics to headlines,
    best_indexes = match_headline_types(self.hadlines_factory.headlines, embedded_headlines, 'headline.', 1.4)

    # 4. find sections
    sections = find_sections_by_headlines(best_indexes, _charter_doc)

    if 'name' in sections:
      section: HeadlineMeta = sections['name']
      org = detect_ners(section.body, context=self, render=verbose)
    else:
      org = {
        'type': 'org_unknown',
        'name': "не определено",
        'type_name': "не определено",
        'tokens': [],
        'attention_vector': []
      }

    rz = find_contraints(sections, self, verbose)

    #   html = render_constraint_values(rz)
    #   display(HTML(html))
    self.org = org
    self.constraints = rz
    self.doc=_charter_doc

    return org, rz


class CharterHeadlinesPatternFactory(AbstractPatternFactory):

  def create_pattern(self, pattern_name, ppp):
    _ppp = (ppp[0].lower(), ppp[1].lower(), ppp[2].lower())
    fp = FuzzyPattern(_ppp, pattern_name)
    self.patterns.append(fp)
    self.patterns_dict[pattern_name] = fp
    return fp

  def __init__(self, embedder):
    AbstractPatternFactory.__init__(self, embedder)
    self.patterns_dict = {}
    self._build_head_patterns()
    self.embedd()

    self.headlines = ['head.directors', 'head.all', 'head.gen', 'head.pravlenie', 'name']

  def _build_head_patterns(self):
    def cp(name, tuples):
      return self.create_pattern(name, tuples)

    head_prfx = "статья 0"

    cp('headline.name.1', ('Полное', 'фирменное наименование', 'общества на русском языке:'))
    cp('headline.name.2', ('', 'ОБЩИЕ ПОЛОЖЕНИЯ', ''))
    cp('headline.name.3', ('', 'фирменное', ''))
    cp('headline.name.4', ('', 'русском', ''))
    cp('headline.name.5', ('', 'языке', ''))
    cp('headline.name.6', ('', 'полное', ''))

    cp('headline.head.all.1', (head_prfx, 'компетенции общего собрания акционеров', ''))
    cp('headline.head.all.1', (head_prfx, 'компетенции общего собрания участников', 'общества'))
    cp('headline.head.all.2', (head_prfx, 'собрание акционеров\n', ''))

    cp('headline.head.all.3', ('', 'компетенции', ''))
    cp('headline.head.all.4', ('', 'собрания', ''))
    cp('headline.head.all.5', ('', 'участников', ''))
    cp('headline.head.all.6', ('', 'акционеров', ''))

    cp('headline.head.directors.1', (head_prfx, 'компетенция совета директоров', 'общества'))
    cp('headline.head.directors.2', ('', 'совет директоров общества', ''))
    cp('headline.head.directors.3', ('', 'компетенции', ''))
    cp('headline.head.directors.4', ('', 'совета', ''))
    cp('headline.head.directors.5', ('', 'директоров', ''))

    cp('headline.head.pravlenie.1', (head_prfx, 'компетенции правления', ''))
    cp('headline.head.pravlenie.2', ('', 'компетенции', ''))
    cp('headline.head.pravlenie.3', ('', 'правления', ''))
    #     cp('d_head_pravlenie.2', ('', 'общества', ''))

    cp('headline.head.gen.1', (head_prfx, 'компетенции генерального директора', ''))
    cp('headline.head.gen.2', ('', 'компетенции', ''))
    cp('headline.head.gen.3', ('', 'генерального', ''))
    cp('headline.head.gen.4', ('', 'директора', ''))


class CharterConstraintsPatternFactory(AbstractPatternFactory):

  def create_pattern(self, pattern_name, ppp):
    _ppp = (ppp[0].lower(), ppp[1].lower(), ppp[2].lower())
    fp = FuzzyPattern(_ppp, pattern_name)
    self.patterns.append(fp)
    self.patterns_dict[pattern_name] = fp
    return fp

  def __init__(self, embedder):
    AbstractPatternFactory.__init__(self, embedder)

    self.patterns_dict = {}

    self._build_order_patterns()
    self._build_sum_margin_extraction_patterns()
    self._build_sum_patterns()
    self.embedd()

  def _build_order_patterns(self):
    def cp(name, tuples):
      return self.create_pattern(name, tuples)

    prefix = 'принятие решения о согласии на совершение или о последующем одобрении'

    cp('d_order_4', (prefix, 'cделки', ', стоимость которой равна или превышает'))
    cp('d_order_5', (prefix, 'cделки', ', стоимость которой составляет менее'))

  def _build_sum_patterns(self):
    def cp(name, tuples):
      return self.create_pattern(name, tuples)

    suffix = 'млн. тыс. миллионов тысяч рублей долларов копеек евро'
    prefix = 'решений о совершении сделок '

    cp('sum_max1', (prefix + 'стоимость', 'не более 0', suffix))
    cp('sum_max2', (prefix + 'цена', 'не больше 0', suffix))
    cp('sum_max3', (prefix + 'стоимость <', '0', suffix))
    cp('sum_max4', (prefix + 'цена менее', '0', suffix))
    cp('sum_max5', (prefix + 'стоимость не может превышать', '0', suffix))
    cp('sum_max6', (prefix + 'общая сумма может составить', '0', suffix))
    cp('sum_max7', (prefix + 'лимит соглашения', '0', suffix))
    cp('sum_max8', (prefix + 'верхний лимит стоимости', '0', suffix))
    cp('sum_max9', (prefix + 'максимальная сумма', '0', suffix))

  def _build_sum_margin_extraction_patterns(self):
    suffix = 'млн. тыс. миллионов тысяч рублей долларов копеек евро'
    prefix = 'совершение сделок '

    # less than
    self.create_pattern('sum__lt_1', (prefix + 'стоимость', 'не более 0', suffix))
    self.create_pattern('sum__lt_2', (prefix + 'цена', 'не больше 0', suffix))
    self.create_pattern('sum__lt_3', (prefix + 'стоимость', '< 0', suffix))
    self.create_pattern('sum__lt_4', (prefix + 'цена', 'менее 0', suffix))
    self.create_pattern('sum__lt_4.1', (prefix + 'цена', 'ниже 0', suffix))
    self.create_pattern('sum__lt_5', (prefix + 'стоимость', 'не может превышать 0', suffix))
    self.create_pattern('sum__lt_6', (prefix + 'лимит соглашения', '0', suffix))
    self.create_pattern('sum__lt_7', (prefix + 'верхний лимит стоимости', '0', suffix))
    self.create_pattern('sum__lt_8', (prefix, 'максимум 0', suffix))
    self.create_pattern('sum__lt_9', (prefix, 'до 0', suffix))
    self.create_pattern('sum__lt_10', (prefix, 'но не превышающую 0', suffix))
    self.create_pattern('sum__lt_11', (prefix, 'совокупное пороговое значение 0', suffix))

    # greather than
    self.create_pattern('sum__gt_1', (prefix + 'составляет', 'более 0', suffix))
    self.create_pattern('sum__gt_2', (prefix + '', 'превышает 0', suffix))
    self.create_pattern('sum__gt_3', (prefix + '', 'свыше 0', suffix))
    self.create_pattern('sum__gt_4', (prefix + '', 'сделка имеет стоимость, равную или превышающую 0', suffix))


class CharterPatternFactory(AbstractPatternFactory):

  @deprecated
  def create_pattern(self, pattern_name, ppp):
    _ppp = (ppp[0].lower(), ppp[1].lower(), ppp[2].lower())
    fp = FuzzyPattern(_ppp, pattern_name)
    self.patterns.append(fp)
    self.patterns_dict[pattern_name] = fp
    return fp

  def __init__(self, embedder):
    AbstractPatternFactory.__init__(self, embedder)

    self.patterns_dict = {}

    #     self._build_paragraph_split_pattern()
    self._build_order_patterns()
    self.embedd()

  @deprecated
  def _build_order_patterns(self):
    def cp(name, tuples):
      return self.create_pattern(name, tuples)

    cp('d_order_1', ('Порядок', 'одобрения сделок', 'в совершении которых имеется заинтересованность'))
    cp('d_order_2', ('', 'принятие решений', 'о совершении сделок'))
    cp('d_order_3',
       ('', 'одобрение заключения', 'изменения или расторжения какой-либо сделки Общества'))
    cp('d_order_4', ('', 'Сделки', 'стоимость которой равна или превышает'))
    cp('d_order_5', ('', 'Сделки', 'стоимость которой составляет менее'))


# self.headlines = ['head.directors', 'head.all', 'head.gen', 'head.pravlenie', 'name']

head_types = ['head.directors', 'head.all', 'head.gen', 'head.pravlenie']

head_types_dict = {'head.directors': 'Совет директоров',
                   'head.all': 'Общее собрание участников/акционеров',
                   'head.gen': 'Генеральный директор',
                   #                      'shareholders':'Общее собрание акционеров',
                   'head.pravlenie': 'Правление общества',
                   'head.unknown': '*Неизвестный орган управления*'}

head_types_colors = {'head.directors': 'crimson',
                     'head.all': 'orange',
                     'head.gen': 'blue',
                     'head.shareholders': '#666600',
                     'head.pravlenie': '#0099cc',
                     'head.unknown': '#999999'}

org_types = {
  'org_unknown': 'undefined',
  'org_ao': 'Акционерное общество',
  'org_zao': 'Закрытое акционерное общество',
  'org_oao': 'Открытое акционерное общество',
  'org_ooo': 'Общество с ограниченной ответственностью'}


class CharterNerPatternFactory(AbstractPatternFactory):

  def create_pattern(self, pattern_name, ppp):
    _ppp = (ppp[0].lower(), ppp[1].lower(), ppp[2].lower())
    fp = FuzzyPattern(_ppp, pattern_name)
    self.patterns.append(fp)
    self.patterns_dict[pattern_name] = fp
    return fp

  def __init__(self, embedder):
    AbstractPatternFactory.__init__(self, embedder)

    self.patterns_dict = {}

    self._build_ner_patterns()
    self.embedd()

  def _build_ner_patterns(self):
    def cp(name, tuples):
      return self.create_pattern(name, tuples)

    for o_type in org_types.keys():
      cp(o_type, ('', org_types[o_type], '"'))

    cp('ner_org.1', ('Полное', 'фирменное наименование', 'общества на русском языке:'))

    cp('ner_org.6', ('', 'ОБЩИЕ ПОЛОЖЕНИЯ', ''))

    cp('ner_org.2', ('', 'фирменное', ''))
    cp('ner_org.3', ('', 'русском', ''))
    cp('ner_org.4', ('', 'языке', ''))
    cp('ner_org.5', ('', 'полное', ''))

    cp('nerneg_1', ('общество имеет', 'печать', ''))
    cp('nerneg_2', ('', 'сокращенное', ''))
    cp('nerneg_3', ('на', 'английском', 'языке'))


def _build_org_type_attention_vector(subdoc: CharterDocument):
  attention_vector_neg = make_soft_attention_vector(subdoc, 'nerneg_1', blur=80)
  attention_vector_neg = 1 + (1 - attention_vector_neg)  # normalize(attention_vector_neg * -1)
  return attention_vector_neg


# ------------------------------------------------------------------------------
def _detect_org_type_and_name(section, render=False):
  s_attention_vector_neg = _build_org_type_attention_vector(section)

  dict_org = {}
  best_type = None
  _max = 0
  for org_type in org_types.keys():

    vector = section.distances_per_pattern_dict[org_type] * s_attention_vector_neg
    if render:
      print('_detect_org_type_and_name, org_type=', org_type, section.distances_per_pattern_dict[org_type][0:10])

    idx = np.argmax(vector)
    val = section.distances_per_pattern_dict[org_type][idx]
    if val > _max:
      _max = val
      best_type = org_type

    dict_org[org_type] = [idx, val]

  if render:
    print('_detect_org_type_and_name', dict_org)

  return dict_org, best_type


# ------------------------------------------------------------------------------
def detect_ners(section, context: CharterAnlysingContext, render=False):
  assert section is not None

  section.embedd(context.ner_factory)
  section.calculate_distances_per_pattern(context.ner_factory)

  dict_org, best_type = _detect_org_type_and_name(section, render)

  if render:
    context.renderer.render_color_text(section.tokens_cc, section.distances_per_pattern_dict[best_type], _range=[0, 1])

  start = dict_org[best_type][0]
  start = start + len(context.ner_factory.patterns_dict[best_type].embeddings)
  end = 1 + find_ner_end(section.tokens, start)

  orgname_sub_section: LegalDocument = section.subdoc(start, end)
  org_name = orgname_sub_section.untokenize_cc

  if render:
    context.renderer.render_color_text(orgname_sub_section.tokens_cc,
                                       orgname_sub_section.distances_per_pattern_dict[best_type],
                                       _range=[0, 1])
    print('Org type:', org_types[best_type], dict_org[best_type])

  rez = {
    'type': best_type,
    'name': org_name,
    'type_name': org_types[best_type],
    'tokens': section.tokens_cc,
    'attention_vector': section.distances_per_pattern_dict[best_type]
  }

  return rez


def _extract_constraint_values_from_region(sentenses_i, _embedd_factory, context: CharterAnlysingContext, render=False):
  if sentenses_i is None or len(sentenses_i) == 0:
    return []

  ssubdocs = embedd_generic_tokenized_sentences(sentenses_i, _embedd_factory)

  for ssubdoc in ssubdocs:

    vectors = make_constraints_attention_vectors(ssubdoc)
    ssubdoc.distances_per_pattern_dict = {**ssubdoc.distances_per_pattern_dict, **vectors}

    if render:
      context.renderer.render_color_text(
        ssubdoc.tokens,
        ssubdoc.distances_per_pattern_dict['deal_value_attention_vector'], _range=(0, 1))

  sentences = []
  for sentence_subdoc in ssubdocs:
    constraints: List[ValueConstraint] = extract_all_contraints_from_sentence(sentence_subdoc,
                                                                              sentence_subdoc.distances_per_pattern_dict[
                                                                                'deal_value_attention_vector'])

    sentence = {
      'quote': untokenize(sentence_subdoc.tokens_cc),
      'subdoc': sentence_subdoc,
      'constraints': constraints
    }

    sentences.append(sentence)
  return sentences


##---------------------------------------
def extract_constraint_values_from_section(section: HeadlineMeta, context: CharterAnlysingContext, verbose=False):
  _embedd_factory = context.price_factory

  if verbose:
    print('extract_constraint_values_from_section', section.type)

  body = section.body

  if verbose:
    print('extract_constraint_values_from_section', 'embedding....')

  sentenses_i = []
  senetences = split_by_token(body.tokens, '\n')
  for s in senetences:
    line = untokenize(s) + '\n'
    sum = extract_sum(line)
    if sum is not None:
      sentenses_i.append(line)
    if verbose:
      print('-', sum, line)

  hl_subdoc = section.subdoc

  r_by_head_type = {
    'section': head_types_dict[section.type],
    'caption': untokenize(hl_subdoc.tokens_cc),
    'sentences': _extract_constraint_values_from_region(sentenses_i, _embedd_factory, context, render=verbose)
  }

  return r_by_head_type


##---------------------------------------
def extract_constraint_values_from_sections(sections, context: CharterAnlysingContext, verbose=False):
  rez = {}

  for head_type in sections:
    section = sections[head_type]
    rez[head_type] = extract_constraint_values_from_section(section, context, verbose)

  return rez


# ---------------------------------------
def find_contraints(sections, context: CharterAnlysingContext, verbose=False):
  # 5. extract constraint values
  sections_filtered = {}
  prefix = 'head.'
  for k in sections:
    if k[:len(prefix)] == prefix:
      sections_filtered[k] = sections[k]

  rz = extract_constraint_values_from_sections(sections_filtered, context, verbose)
  return rz


# ------------------------------