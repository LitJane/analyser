from typing import List

import numpy as np

from legal_docs import LegalDocument, deprecated, HeadlineMeta, get_sentence_bounds_at_index
from ml_tools import put_if_better, cut_above, relu, filter_values_by_key_prefix, \
  rectifyed_sum, smooth_safe
from parsing import ParsingContext, ParsingSimpleContext
from patterns import AbstractPatternFactory


class SectionsFinder:

  def __init__(self, ctx: ParsingSimpleContext):
    self.ctx: ParsingSimpleContext = ctx
    pass

  def find_sections(self, doc: LegalDocument, factory: AbstractPatternFactory, headlines: List[str],
                    headline_patterns_prefix: str = 'headline.', additional_attention: List[float] = None) -> dict:
    raise NotImplementedError()


class DefaultSectionsFinder(SectionsFinder):
  def __init__(self, ctx: ParsingContext):
    SectionsFinder.__init__(self, ctx)

  @deprecated
  def find_sections(self, doc: LegalDocument, factory: AbstractPatternFactory, headlines: List[str],
                    headline_patterns_prefix: str = 'headline.', additional_attention: List[float] = None) -> dict:
    embedded_headlines = doc.embedd_headlines(factory)

    doc.sections = doc.find_sections_by_headlines_2(
      self.ctx, headlines, embedded_headlines, headline_patterns_prefix, self.ctx.config.headline_attention_threshold)

    self.ctx._logstep("embedding headlines into semantic space")

    return doc.sections


class FocusingSectionsFinder(SectionsFinder):
  def __init__(self, ctx: ParsingSimpleContext):
    SectionsFinder.__init__(self, ctx)

  def find_sections(self, doc: LegalDocument, factory: AbstractPatternFactory, headlines: List[str],
                    headline_patterns_prefix: str = 'headline.', additional_attention: List[float] = None) -> dict:

    """
    Fuzziy Finds sections in the doc
    TODO: try it on Contracts and Protocols as well
    TODO: if well, move from here

    🍄 🍄 🍄 🍄 🍄 Keep in in the dark and feed it sh**



    """

    def is_hl_more_confident(a: HeadlineMeta, b: HeadlineMeta):
      return a.confidence > b.confidence

    #     assert do
    headlines_attention_vector = self.normalize_headline_attention_vector(self.make_headline_attention_vector(doc))

    section_by_index = {}
    for section_type in headlines:
      # like ['name.', 'head.all.', 'head.gen.', 'head.directors.']:
      pattern_prefix = f'{headline_patterns_prefix}{section_type}'
      doc.calculate_distances_per_pattern(factory, pattern_prefix=pattern_prefix, merge=True)

      # warning! these are the boundaries of the headline, not of the entire section
      bounds, confidence, attention = self._find_charter_section_start(doc, pattern_prefix, headlines_attention_vector,
                                                                       additional_attention)
      sl = slice(bounds[0], bounds[1])
      hl_info = HeadlineMeta(None, section_type, confidence, doc.subdoc_slice(sl, name=section_type))
      hl_info.attention = attention
      put_if_better(section_by_index, sl.start, hl_info, is_hl_more_confident)
    # end-for
    # s = slice(bounds[0], bounds[1])

    sorted_starts = [i for i in sorted(section_by_index.keys())]
    sorted_starts.append(len(doc.tokens))
    section_by_type = {}
    for i in range(len(sorted_starts) - 1):
      index = sorted_starts[i]
      section: HeadlineMeta = section_by_index[index]
      start = index  # todo: probably take the end of the caption
      end = sorted_starts[i + 1]

      section_len = end - start
      if section_len > 5000:
        self.ctx.warning(
          f'Section "{section.subdoc.untokenize_cc()[:100]}" is probably way too large {section_len}, timming to 5000 ')
        section_len = 5000  #

      sli = slice(start, start + section_len)
      section.body = doc.subdoc_slice(sli, name=section.type)
      section.attention = section.attention[sli]
      section_by_type[section.type] = section

    # end-for
    doc.sections = section_by_type
    return section_by_type

  """ ❤️ == GOOD HEART LINE ====================================================== """

  def make_headline_attention_vector(self, doc):
    level_by_line = [max(i._possible_levels) for i in doc.structure.structure]

    headlines_attention_vector = []
    for i in doc.structure.structure:
      l = i.span[1] - i.span[0]
      headlines_attention_vector += [level_by_line[i.line_number]] * l

    return np.array(headlines_attention_vector)

  """ ❤️ == GOOD HEART LINE ====================================================== """

  def normalize_headline_attention_vector(self, headline_attention_vector_pure):
    # XXX: test it
    #   _max_head_threshold = max(headline_attention_vector_pure) * 0.75
    _max_head_threshold = 1  # max(headline_attention_vector_pure) * 0.75
    # XXX: test it
    #   print(_max_head)
    headline_attention_vector = cut_above(headline_attention_vector_pure, _max_head_threshold)
    #   headline_attention_vector /= 2 # 5 is the maximum points a headline may gain during headlne detection : TODO:
    return relu(headline_attention_vector)

  """ ❤️ == GOOD HEART LINE ====================================================== """

  def _find_charter_section_start(self, doc, headline_pattern_prefix, headlines_attention_vector, additional_attention):

    assert headlines_attention_vector is not None

    vectors = filter_values_by_key_prefix(doc.distances_per_pattern_dict, headline_pattern_prefix)
    v = rectifyed_sum(vectors, 0.3)

    if additional_attention is not None:
      additional_attention_s = smooth_safe(additional_attention, 6)
      v += additional_attention_s

    v *= headlines_attention_vector

    span = 100
    best_id = np.argmax(v)
    dia = slice(max(0, best_id - span), min(best_id + span, len(v)))
    # debug_renderer(headline_pattern_prefix, self.doc.tokens_cc[dia], normalize(v[dia]))

    bounds = get_sentence_bounds_at_index(best_id, doc.tokens)
    confidence = v[best_id]
    return bounds, confidence, v