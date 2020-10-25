import unittest
import numpy as np
import supportvectormachine as svm


class SupportVectorMachineTests(unittest.TestCase):

    def test_compute_loss_2inputrecords_3outputlabel(self):

        input = np.array([[1, 1, 1],
                          [2, 2, 2]])
        weights = np.array([[0.1, 0.1, 0.1],
                            [0.2, 0.2, 0.2],
                            [0.4, 0.4, 0.4]])
        y = np.array([[1, 0, 0],
                      [0, 0, 1]])

        # result = Input * weights
        # [[0.3, 0.6, 1.2],
        #  [0.6, 1.2, 2.4]]

        # result - result_of_correct_label_for_that_row = result2
        # [[nan, 0.6, 1.2],  -  [[0.3, 0.3, 0.3],   = [[ nan, 0.3,  0.9],
        #  [0.6, 1.2, nan]]      [2.4, 2.4, 2.4]]     [-1.8, -1.2, nan]]

        # max(0, result2 + delta=0.1) = result3
        # [[nan, 0.4, 1.0],
        #  [0,   0,   nan]]

        # sum(result3) = 0.4 + 1.0 = 1.4

        result = svm.SupportVectorMachine.compute_loss(input, weights, y, delta=0.1)

        assert 1.3999 < result < 1.40001

    def test_compute_loss_2inputrecords_3outputlabel(self):

        input = np.array([[1, 1, 1],
                          [2, 2, 2]])
        weights = np.array([[0.1, 0.1, 0.1],
                            [0.2, 0.2, 0.2],
                            [0.4, 0.4, 0.4]])
        y = np.array([[1, 0, 0],
                      [0, 0, 1]])

        #  diff
        # [[nan, 1  , 1],
        #  [0,   0, nan]]

        # number_of_labels_predicted_wrong
        # (2)
        # (0)

        # gradient-1
        # [-2, -2, -2]
        # [0, 0, 0]
        # [0, 0, 0]

        # gradient-2
        # [-2, -2, -2]
        # [1, 1, 1]
        # [1, 1, 1]

        hinge_loss_term = svm.SupportVectorMachine.compute_hinge_loss_term(
            input, weights, y, delta=0.1)

        result = svm.SupportVectorMachine.compute_gradient_analitically(
            input, weights, y, hinge_loss_term,regularization_value=0)

        expected = np.array([[-2, -2, -2],
                             [1, 1, 1],
                             [1, 1, 1]])
        assert np.array_equal(result, expected)

    def test_gaussian_kernel(self):

        input = np.array([[0.1, 0.2, 0.3],
                          [10, 20, 30],  # high values so the result is close to 0
                          [0.1, 0.5, 0.7]])

        result = svm.SupportVectorMachine.apply_gaussian_kernel(input, gamma=0.3)

        assert np.all(0 < result[0,2] < 1) and \
                np.all(0 <= result[1]) and np.all(result[1] < 0.00001)

