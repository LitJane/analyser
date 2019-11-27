from abc import abstractmethod

import numpy as np

from analyser.documents import TextMap
from analyser.text_tools import Tokens


def embedd_tokenized_sentences_list(embedder, tokenized_sentences_list):
  maxlen = 0
  lens = []

  for s in tokenized_sentences_list:
    lens.append(len(s))
    if len(s) > maxlen:
      maxlen = len(s)

  _strings = []
  for i in range(len(tokenized_sentences_list)):
    s = tokenized_sentences_list[i]
    s.extend([' '] * (maxlen - len(s)))
    _strings.append(s)

  _strings = np.array(_strings)

  # ======== call TENSORFLOW -----==================
  sentences_emb = embedder.embedd_tokenized_text(_strings, lens)
  # ================================================
  return sentences_emb, None, lens


class AbstractEmbedder:

  @abstractmethod
  def get_embedding_tensor(self, tokenized_sentences_list):
    raise NotImplementedError()

  @abstractmethod
  def embedd_tokens(self, tokens: Tokens) -> np.ndarray:
    raise NotImplementedError()

  @abstractmethod
  def embedd_tokenized_text(self, words: [Tokens], lens: [int]) -> np.ndarray:
    raise NotImplementedError()

  @abstractmethod
  def embedd_strings(self, strings: Tokens) -> np.ndarray:
    raise NotImplementedError()

  def embedd_contextualized_patterns(self, patterns, trim_padding=True):

    tokenized_sentences_list: [Tokens] = []
    regions = []

    i = 0
    maxlen = 0
    lens = []
    for (ctx_prefix, pattern, ctx_postfix) in patterns:
      # sentence = ' '.join((ctx_prefix, pattern, ctx_postfix))

      prefix_tokens = TextMap(ctx_prefix).tokens  # tokenize_text(ctx_prefix)
      pattern_tokens = TextMap(pattern).tokens
      suffix_tokens = TextMap(ctx_postfix).tokens

      start = len(prefix_tokens)
      end = start + len(pattern_tokens)

      sentence_tokens = prefix_tokens + pattern_tokens + suffix_tokens

      # print('embedd_contextualized_patterns', (sentence, start, end))

      regions.append((start, end))
      tokenized_sentences_list.append(sentence_tokens)
      lens.append(len(sentence_tokens))
      if len(sentence_tokens) > maxlen:
        maxlen = len(sentence_tokens)

      i = i + 1

    # print('maxlen=', maxlen)
    _strings = []

    for s in tokenized_sentences_list:
      s.extend([' '] * (maxlen - len(s)))
      _strings.append(s)
      # print(s)
    _strings = np.array(_strings)

    # ======== call TENSORFLOW -----==================
    sentences_emb = self.embedd_tokenized_text(_strings, lens)
    # ================================================

    # print(sentences_emb.shape)
    #     assert len(sentence_tokens) == sentences_emb

    patterns_emb = []

    if trim_padding:
      for i in range(len(regions)):
        start, end = regions[i]

        sentence_emb = sentences_emb[i]
        pattern_emb = sentence_emb[start:end]

        patterns_emb.append(pattern_emb)
      patterns_emb = np.array(patterns_emb)
    else:
      patterns_emb = sentences_emb

    return patterns_emb, regions
