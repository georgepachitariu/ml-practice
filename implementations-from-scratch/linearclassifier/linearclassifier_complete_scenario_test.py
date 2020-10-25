from linearclassifier import LinearClassifier
import unittest
import numpy as np
import dataset.dataset as d


class LinearClassifierExercises(unittest.TestCase):
    """
    Here there are exercises/tests that a network needs to be able to learn/train.
    (You can see them also as integration/scenario tests)
    """
    @staticmethod
    def build_train_and_test(x_train, y_train, with_bias):
        x_train, _, _ = d.Dataset().normalize(x_train)
        y_train = d.Dataset.transform_y_from_label_values_to_label_indices(y_train, 2)

        classifier = LinearClassifier([x_train.shape[1], y_train.shape[1]], with_bias=with_bias,
                                      regularization_value=0.01, random_seed=0)

        classifier.train(x_train, y_train, learning_rate=0.1, iterations=500, debug_enabled=True)

        # the network should be able predict all train cases correctly
        predicted = classifier.predict(x_train)
        _, percentage = classifier.get_predicted_correctly(predicted, y_train)

        return percentage

    def test_function1_no_bias(self):
        #   f(x) = 1 when x > 0
        #   f(x) = 0 when x <= 0
        x_train = np.array([[-1],  [1]])
        y_train = np.array([  0,    1 ])
        percentage = LinearClassifierExercises.build_train_and_test(x_train, y_train, False)

        assert percentage == 100

    def test_function2_with_bias(self):
        #   f(x) = 1 when x > 3
        #   f(x) = 0 when x <= 3
        x_train = np.array([[2],  [5]])
        y_train = np.array([ 0,    1 ])
        percentage = LinearClassifierExercises.build_train_and_test(x_train, y_train, True)
        assert percentage == 100

    def test_function2_with_bias(self):
        #   f1(0) = 1
        x_train = np.array([[0]])
        y_train = np.array([ 1 ])
        percentage = LinearClassifierExercises.build_train_and_test(x_train, y_train, False)
        # without bias term it shouldn't be able to fit the case
        assert percentage == 0
