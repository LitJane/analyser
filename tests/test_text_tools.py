import unittest

from documents import TextMap
from text_tools import find_best_sentence_end, split_into_sentences


class TextToolsTestCase(unittest.TestCase):

  def test_split_into_sentences(self):
    s = 'Предложение 1.Предложение 2.'
    spans = split_into_sentences(s, 20)

    self.assertEqual('Предложение 2.', s[spans[1][0]:spans[1][1]])
    self.assertEqual('Предложение 1.', s[spans[0][0]:spans[0][1]])

  def test_split_into_sentences_0(self):
    s = 'Предложение 1. Предложение 2.'
    spans = split_into_sentences(s, 20)

    self.assertEqual(' Предложение 2.', s[spans[1][0]:spans[1][1]])
    self.assertEqual('Предложение 1.', s[spans[0][0]:spans[0][1]])

  def test_find_best_sentence_end_0(self):
    x = 'ие? д'
    be = find_best_sentence_end(x)
    self.assertEqual("ие?", x[:be])

  def test_find_best_sentence_end_1(self):
    x = 'ие. д'
    be = find_best_sentence_end(x)
    self.assertEqual("ие.", x[:be])

  def test_find_best_sentence_end_2(self):
    x = 'ие) д'
    be = find_best_sentence_end(x)
    self.assertEqual("ие)", x[:be])

  def test_find_best_sentence_end_3(self):
    x = 'ие) д!'
    be = find_best_sentence_end(x)
    self.assertEqual("ие) д!", x[:be])

  def test_find_best_sentence_end_4(self):
    x = 'иед'
    be = find_best_sentence_end(x)
    self.assertEqual("иед", x[:be])

  def test_find_best_sentence_end_5(self):
    x = 'иед: нет'
    be = find_best_sentence_end(x)
    self.assertEqual("иед:", x[:be])


def split_sentences_into_map(substr, max_len_chars=150) -> TextMap:
  spans1 = split_into_sentences(substr, max_len_chars)
  tm = TextMap(substr, spans1)
  return tm


if __name__ == '__main__':
  unittest.main()
