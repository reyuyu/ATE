import unittest
import numpy as np
from .analyze import boot_mean


class StatisticsTests(unittest.TestCase):
    def test_question_weighting_and_paired_zero(self):
        samples = [{'image_sha256': h} for h in ('a', 'a', 'a', 'b')]
        values = np.array([[1, 1], [1, 1], [1, 1], [0, 0]], float)
        mean, low, high, draws, groups = boot_mean(values, samples, repetitions=1000)
        self.assertEqual(groups, 2)
        np.testing.assert_array_equal(mean, [.75, .75])
        np.testing.assert_array_equal(draws[:, 0]-draws[:, 1], np.zeros(1000))

    def test_six_condition_matrix_indices(self):
        scores = np.array([[1, 0, 1, 1, 0, 0], [0, 1, 0, 1, 1, 1]], float)
        matrix = np.concatenate([scores*100, (scores[:, 1:]-scores[:, :1])*100,
            np.stack([(scores[:, j]-scores[:, 1])*100 for j in (2, 3, 4)], axis=1)], axis=1)
        for j in range(1, 6):
            np.testing.assert_array_equal(matrix[:, 5+j], (scores[:, j]-scores[:, 0])*100)
        for k, j in enumerate((2, 3, 4)):
            np.testing.assert_array_equal(matrix[:, 11+k], (scores[:, j]-scores[:, 1])*100)


if __name__ == '__main__':
    unittest.main()
