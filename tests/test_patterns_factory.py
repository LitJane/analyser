import unittest

import numpy as np

from embedding_tools import AbstractEmbedder
from legal_docs import *


class FakeEmbedder(AbstractEmbedder):

  def __init__(self, default_point):
    self.default_point = default_point

  def embedd_tokenized_text(self, tokenized_sentences_list, lens):
    # def get_embedding_tensor(self, tokenized_sentences_list):
    tensor = []
    for sent in tokenized_sentences_list:
      sentense_emb = []
      for token in sent:
        token_emb = self.default_point
        sentense_emb.append(token_emb)
      tensor.append(sentense_emb)

    return np.array(tensor), tokenized_sentences_list


class FakePatternFactory(AbstractPatternFactory):
  pass
  # def __init__(self, embedder):
  #     super(embedder)


class CoumpoundFuzzyPatternTestCase(unittest.TestCase):

  # def __init__(self):

  def test_embedder(self):
    point1 = [1, 6, 4]

    PF = AbstractPatternFactory(FakeEmbedder(point1))

    fp1 = PF.create_pattern('p1', ('prefix', 'pat 2', 'suffix'))
    fp2 = PF.create_pattern('p2', ('prefix', 'pat', 'suffix 2'))
    fp3 = PF.create_pattern('p3', ('', 'a b c', ''))

    self.assertEqual(3, len(PF.patterns))

    PF.embedd()

    self.assertEqual(2, len(fp1.embeddings))
    self.assertEqual(1, len(fp2.embeddings))
    self.assertEqual(3, len(fp3.embeddings))

  def test_coumpound_find(self):
    point1 = [1, 3, 7]
    point2 = [1, 7, 4]
    point3 = [1, 6, 4]

    embedder = FakeEmbedder(point2)
    PF = AbstractPatternFactory(embedder)

    fp1 = PF.create_pattern('p1', ('prefix', 'pat', 'suffix 1'))
    fp2 = PF.create_pattern('p2', ('prefix', 'pat', 'suffix 2'))

    PF.embedd()
    # FuzzyPattern(None)
    # fp2.set_embeddings(np.array([point2]))
    # fp2 = FuzzyPattern(np.array([[point2]]))

    cp = CoumpoundFuzzyPattern()
    cp.add_pattern(fp1, 0.5)
    cp.add_pattern(fp2, 0.5)
    # cp.add_pattern(fp2, 2)

    text_emb = np.array([point1, point2, point3, point3])
    min_i, sums, confidence = cp.find(text_emb)

    print(min_i, sums, confidence)

    self.assertEqual(1, min_i)


if __name__ == '__main__':
  unittest.main()
