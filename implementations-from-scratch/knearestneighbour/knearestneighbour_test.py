import unittest
from knearestneighbour import KNearestNeighbour
import numpy as np


class KNearestNeighbourTests(unittest.TestCase):

    def test_simple(self):
        a = np.array([1, 2, 1])
        b = np.array([[1, 1, 3]])
        result = KNearestNeighbour().computeEuclideanDistance(a, b)

        # sqrt( (1-1)^2 + (2-1)^2 + (1-3)^2 ) = sqrt(5) ~= 2.236
        assert 0 < result - 2.236 < 0.0001

    def test_getEuclideandistanceBetweenEachTrainAndTestRow(self):
        train = np.array([[1, 2, 1],
                      [1, 2, 2],
                      [2, 2, 1]])
        test = np.array([[1, 1, 1],
                      [2, 2, 2]])

        expected = np.array([[1, 1.41, 1.41],
                             [1.41, 1, 1]])

        result = KNearestNeighbour().getEuclideandistanceBetweenEachTrainAndTestRow(train, test)
        assert np.allclose(expected, result, rtol=0.01)

    def test_predict(self):
        train_x = np.array([[1, 1, 1],
                            [2, 2, 2]])
        train_y = np.array([5, 6])

        test_x = np.array([[2, 2, 1],
                           [1, 2, 1]])

        expected = np.array([6, 5])

        result = KNearestNeighbour().predict(train_x, train_y, test_x)
        assert np.all(result == expected)

if __name__ == '__main__':
    unittest.main()