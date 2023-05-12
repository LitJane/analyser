import os
from abc import abstractmethod

import numpy as np

from analyser.documents import TextMap
from analyser.hyperparams import datasets_dir
from analyser.log import logger as elmo_logger
from analyser.ml_tools import Embeddings
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

  def __cache_fn(self, checksum):
    return os.path.join(datasets_dir, f'cache-{checksum}-embeddings-ElmoEmbedder.npy')

  def get_cached_embedding(self, checksum) -> Embeddings or None:
    fn = self.__cache_fn(checksum)
    # print(f'checking for existence {fn}')
    if os.path.isfile(fn):
      elmo_logger.debug(f'skipping embedding doc {checksum} ...., {fn} exists, loading')
      e = np.load(fn)
      elmo_logger.debug(f'loaded embedding shape is: {e.shape}')
      return e

    return None

  def cache_embedding(self, checksum, embeddings):
    fn = self.__cache_fn(checksum)
    np.save(fn, embeddings)

  @abstractmethod
  def embedd_tokens(self, tokens: Tokens) -> Embeddings:
    raise NotImplementedError()

  @abstractmethod
  def embedd_tokenized_text(self, words: [Tokens], lens: [int]) -> np.ndarray:
    raise NotImplementedError()

  @abstractmethod
  def embedd_strings(self, strings: Tokens) -> Embeddings:
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

      regions.append((start, end))
      tokenized_sentences_list.append(sentence_tokens)
      lens.append(len(sentence_tokens))
      if len(sentence_tokens) > maxlen:
        maxlen = len(sentence_tokens)

      i = i + 1

    _strings = []

    for s in tokenized_sentences_list:
      s.extend([' '] * (maxlen - len(s)))
      _strings.append(s)
    _strings = np.array(_strings)

    # ======== call TENSORFLOW -----==================
    sentences_emb = self.embedd_tokenized_text(_strings, lens)
    # ================================================

    patterns_emb = []

    if trim_padding:
      for i, (start, end) in enumerate(regions):
        sentence_emb = sentences_emb[i]
        pattern_emb = sentence_emb[start:end]

        patterns_emb.append(pattern_emb)
      patterns_emb = np.array(patterns_emb)
    else:
      patterns_emb = sentences_emb

    return patterns_emb, regions

  def embedd_large(self, text_map, max_tokens=6000, log_addon=''):
    elmo_logger.info(f'{log_addon} {len(text_map)} max_tokens={max_tokens}')
    overlap = max_tokens // 20

    number_of_windows = 1 + len(text_map) // max_tokens
    window = max_tokens

    msg = f"{log_addon} Document is too large for embedding: {len(text_map)} tokens. Splitting into {number_of_windows} windows overlapping with {overlap} tokens "
    elmo_logger.warning(msg)

    start = 0
    embeddings = None
    # tokens = []
    while start < len(text_map):

      subtokens: Tokens = text_map[start:start + window + overlap]
      elmo_logger.debug(f"{log_addon} Embedding region: {start}, {len(subtokens)}")

      sub_embeddings = self.embedd_tokens(subtokens)[0:window]

      if embeddings is None:
        embeddings = sub_embeddings
      else:
        embeddings = np.concatenate([embeddings, sub_embeddings])

      start += window

    return embeddings
    # self.tokens = tokens
