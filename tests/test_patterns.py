import unittest

from patterns import *


class CoumpoundFuzzyPatternTestCase(unittest.TestCase):

  def test_onehot(self):
    ep = ExclusivePattern()
    a = np.array([[3.0, 2, 3], [2, 3, 5]])
    mask = -np.inf
    m = ep.onehot_column(a, -np.inf)
    print(m)
    self.assertTrue(np.allclose(m, np.array([[3, mask, mask], [mask, 3, 5]])))

  def test_exclusive_find(self):
    point1 = [1, 3]
    point2 = [1, 7]
    point3 = [1, 6]

    fp1 = FuzzyPattern(None)
    fp1.set_embeddings(np.array([point2]))

    fp2 = FuzzyPattern(None)
    fp2.set_embeddings(np.array([point3]))

    cp = ExclusivePattern()
    cp.add_pattern(fp1)
    cp.add_pattern(fp2)

    text_emb = np.array([point1, point2, point3, point3, point1])
    distances_per_pattern, ranges, winning_patterns = cp.calc_exclusive_distances(text_emb)

    print("distances_per_pattern")
    print(distances_per_pattern[0])
    print(distances_per_pattern[1])

    print("winning_patterns")
    print(winning_patterns)

    print("ranges")
    print(ranges)

  def test_eval_distances_soft_pattern2(self):
    point1 = [1, 3]
    point2 = [1, 7]

    point3 = [1, 6]
    point35 = [1, 6.5]

    fp1 = FuzzyPattern(None)
    fp1.set_embeddings(np.array([point3]))

    fp2 = FuzzyPattern(None)
    fp2.set_embeddings(np.array([point35]))

    cp = CoumpoundFuzzyPattern()
    cp.add_pattern(fp2)
    cp.add_pattern(fp1)

    text_emb = np.array([point1, point3, point2, point2])
    sums = cp._find_patterns(text_emb)
    self.assertEqual(len(text_emb), len(sums))

    line0 = sums
    print(line0)

    self.assertGreater(line0[2], line0[1])
    self.assertGreater(line0[0], line0[1])
    self.assertGreater(line0[3], line0[1])

  def test_eval_distances(self):
    point1 = [1, 3]
    point2 = [1, 7]

    embedding_point = [1, 6]

    # fp1 = FuzzyPattern(np.array([[point3], [point2]]))

    fp1 = FuzzyPattern(None)
    fp1.set_embeddings(np.array([embedding_point]))

    text_emb = np.array([point1, point2, embedding_point])
    sums = fp1._eval_distances(text_emb)
    print('sums=', sums)
    print('sums.shape=', sums.shape, 'len of shape=', len(sums.shape))
    self.assertEqual(1, len(sums.shape))
    self.assertEqual(len(text_emb), len(sums))

    line0 = sums
    # print(line0)
    # print(sums[:,1])

    self.assertAlmostEqual(line0[2], 0)

    self.assertGreater(line0[0], line0[1])
    self.assertGreater(line0[1], line0[2])
    self.assertGreater(line0[0], line0[2])

  def test_eval_distances_2(self):
    point1 = [1, 3]
    point2 = [1, 7]

    embedding_point = [1, 6]
    embedding_point2 = [1, 6.01]

    # fp1 = FuzzyPattern(np.array([[point3], [point2]]))

    pattern = FuzzyPattern(None)
    pattern.set_embeddings(np.array([embedding_point, embedding_point2]))

    text_emb = np.array([point1, point2, embedding_point, embedding_point, point2])
    # ----------------
    distances = pattern._eval_distances(text_emb)
    # ----------------
    print('sums=', distances)
    print('sums.shape=', distances.shape, 'len of shape=', len(distances.shape))
    self.assertEqual(1, len(distances.shape))
    self.assertEqual(len(text_emb), len(distances))

    self.assertAlmostEqual(distances[2], 0)

    self.assertGreater(distances[0], distances[1])
    self.assertGreater(distances[1], distances[2])
    self.assertGreater(distances[0], distances[2])

    self.assertEqual(2, np.argmin(distances))

  def test_eval_distances_large_pattern(self):
    point1 = [1, 3]
    point2 = [1, 7]

    embedding_point = [1, 6]
    embedding_point2 = [1, 6.01]

    # fp1 = FuzzyPattern(np.array([[point3], [point2]]))

    pattern = FuzzyPattern(None, _name='test pattern of len 2')
    pattern.set_embeddings(np.array([embedding_point, embedding_point2, embedding_point2, embedding_point2]))

    text_emb = np.array([point1, point2])
    # ----------------
    distances = pattern._eval_distances(text_emb)
    # ----------------
    print('sums=', distances)
    # print ('sums.shape=', distances.shape, 'len of shape=', len(distances.shape))
    # self.assertEqual(1, len(distances.shape))
    # self.assertEqual(len(text_emb), len(distances))
    #
    # self.assertAlmostEqual(distances[2], 0)
    #
    # self.assertGreater(distances[0], distances[1])
    # self.assertGreater(distances[1], distances[2])
    # self.assertGreater(distances[0], distances[2])
    #
    # self.assertEqual(2, np.argmin(distances))

  def test_coumpound_find(self):
    point1 = [1, 3]
    point2 = [1, 7]
    point3 = [1, 6]

    fp2 = FuzzyPattern(None)
    fp2.set_embeddings(np.array([point2]), (0, 1))
    # fp2 = FuzzyPattern(np.array([[point2]]))

    cp = CoumpoundFuzzyPattern()
    cp.add_pattern(fp2, 0.5)
    # cp.add_pattern(fp2, 2)

    text_emb = np.array([point1, point2, point3])
    min_i, sums, confidence = cp.find(text_emb)

    print(min_i, sums, confidence)

    self.assertEqual(1, min_i)

  def test_etimate_confidence(self):
    from ml_tools import estimate_confidence
    confidence, sum_, nonzeros_count, _max = estimate_confidence([])
    self.assertEqual(0, confidence)
    self.assertTrue(sum_ is np.nan)
    self.assertEqual(0, nonzeros_count)
    self.assertTrue(_max is np.nan)

  def test_etimate_confidence2(self):
    from ml_tools import estimate_confidence
    confidence, sum_, nonzeros_count, _max = estimate_confidence([1])
    self.assertEqual(1, confidence)
    self.assertEqual(sum_, 1)
    self.assertEqual(1, nonzeros_count)
    self.assertEqual(_max, 1)


if __name__ == '__main__':
  unittest.main()
