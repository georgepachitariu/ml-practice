import unittest
import numpy as np
from linearclassifier import LinearClassifier


class LinearClassifierTests(unittest.TestCase):

    def test_3nodes_nobias_1record_1iterations_assert_weight_increases(self):
        x_train = np.array([[0.1, 0.1, 0.1,]])
        y_train = np.array([[1]])
        weights = [np.array([[0.1, -0.9, 0.9]])]
        weights_copy = weights[0].copy()
        # x_train * weights = [0.01, -0.09, 0.9]
        # To get to 1, all weights should increase
        net = LinearClassifier(layers_weights=weights, with_bias=False)
        _, _, cost_derivative = net.run_iteration(x_train, y_train)

        # this makes weight value increase
        assert np.all(cost_derivative[0] < 0)

        # assert that all weights values increased
        assert np.all(net.layers_weights[0] - weights_copy > 0)

    def test_1node_nobias_1record_2iterations_oppositeweights(self):
        x_train = np.array([[0.1]])
        y_train = np.array([[1]])
        weights = [np.array([[0.1]])]
        # x_train * weights = 0.01.
        # To get to 1, the weight should increase
        net = LinearClassifier(layers_weights=weights, with_bias=False)

        first_cost, _, _ = net.run_iteration(x_train, y_train, compute_cost=True)
        snd_cost, _, _ = net.run_iteration(x_train, y_train, compute_cost=True)

        # cost should decrease
        assert first_cost - snd_cost > 0

    def test_1nodes_nobias_1record_1iterations_assert_weight_decreases(self):
        x_train = np.array([[0.9]])
        y_train = np.array([[0]])
        weights = [np.array([[0.9]])]
        weights_copy = weights[0].copy()

        net = LinearClassifier(layers_weights=weights, with_bias=False)
        _, _, cost_derivative = net.run_iteration(x_train, y_train)

        # this makes weight value decrease
        assert np.all(cost_derivative[0] > 0)

        # assert that weight value decreased
        assert np.all(net.layers_weights[0] - weights_copy < 0)

    def test_noRegularization_highweights(self):
        # Having high weights means that the network is overfitting the training data
        # and cannot (has no power) generalize well anymore (for unseen data)
        x_train = np.array([[0.5]])
        y_train = np.array([[1]])
        weights = [np.array([[0.7]])]

        net = LinearClassifier(layers_weights=weights, regularization_value=0, with_bias=False)
        for _ in range(1000):
            net.run_iteration(x_train, y_train)

        assert np.all(net.layers_weights[0] > 5)

    def test_regularization_controlledweights(self):
        # same test as above, but with regularization value, that limits (does not allow)
        # for the weight to have a high value
        x_train = np.array([[0.5]])
        y_train = np.array([[1]])
        weights = [np.array([[0.7]])]

        net = LinearClassifier(layers_weights=weights, regularization_value=0.5, with_bias=False)
        for _ in range(1000):
            net.run_iteration(x_train, y_train)

        assert np.all(net.layers_weights[0] < 1)


if __name__ == '__main__':
    unittest.main()
