import neuralnetwork as nn
import unittest
import numpy as np


class Scenario:
    # Also this has the same behaviour as the last layer in a multi-layer network
    def compute(x, y, weights):

        _, activation_value = nn.FeedForward.compute_activation_value(x, weights[0], nn.Sigmoid.compute)

        cost = nn.Cost.compute(activation_value, y, weights, 0)

        layer_error = nn.BackPropagation.compute_last_layer_error_deriv(activation_value, y)

        ## delta TODO
        return activation_value, cost, layer_error

class SimpleScenario_1_trainingexample_1_feature_2_labels(unittest.TestCase):
    def test_predicted_1_real_1(self):
        x = np.array([[0.5]])
        y = np.array([[1]])
        weights = [np.array([[20]])]

        predicted, cost, layer_error = Scenario.compute(x, y, weights)

        # S1. The predicted value is sigmoid(0.5 * 20)~=1
        assert 0.999 < predicted < 1

        # S2. The cost should be small. The cost is always positive
        assert 0 < cost < 0.0001

        # S2. predicted ~= 1 and real = 1. Then the error should be small and negative
        assert -0.0001 < layer_error < 0

    def test_predicted_0_real_0(self):
        x = np.array([[0.5]])
        y = np.array([[0]])
        weights = [np.array([[-20]])]

        predicted, cost, layer_error = Scenario.compute(x, y, weights)

        assert 0 < predicted < 0.0001
        assert 0 < cost < 0.0001
        assert 0 < layer_error < 0.0001

    def test_predicted_1_real_0(self):
        x = np.array([[0.5]])
        y = np.array([[0]])
        weights = [np.array([[20]])]

        predicted, cost, layer_error = Scenario.compute(x, y, weights)

        assert 0.999 < predicted < 1
        assert 1 < cost
        assert 0.999 < layer_error < 1

    def test_predicted_0_real_1(self):
        x = np.array([[0.5]])
        y = np.array([[1]])
        weights = [np.array([[-20]])]

        predicted, cost, layer_error = Scenario.compute(x, y, weights)

        assert 0 < predicted < 0.0001
        assert 1 < cost
        assert -1 < layer_error < -0.999


class Tests_MiddleLayer(unittest.TestCase):
    """
    Feedforward is not tested here, because it has the same behaviour in every layer,
    and it was tested for the last layer above
    """

    def test_normal_case_positive_err(self):
        # -1 < incoming_bp_error < 1 from last layer
        incoming_bp_error=np.array([[0.99]])
        weights = np.array([[0.8]])
        feedforward_mult = np.array([[1.2]])

        outgoing_bp_error = nn.BackPropagation.compute_current_layer_error_deriv(
            incoming_bp_error, weights, feedforward_mult, nn.Sigmoid.derivative, with_bias=False)

        # < 1 doesn't really mean anything
        assert 0 < outgoing_bp_error < 1

    def test_normal_case_negative_err(self):
        incoming_bp_error = np.array([[-0.99]])
        weights = np.array([[10]])
        feedforward_mult = np.array([[0]])  # sigmoid derivative has the highest value in 0

        outgoing_bp_error = nn.BackPropagation.compute_current_layer_error_deriv(
            incoming_bp_error, weights, feedforward_mult, nn.Sigmoid.derivative, with_bias=False)

        assert outgoing_bp_error < -1

    def test_converging_is_impossible(self):
        incoming_bp_error = np.array([[0.99]])
        weights = np.array([[1]])

        # when feedforward_mult is very big or very small, the error is 0,
        # and the current layer cannot converge(train) anymore
        # (also next layers because of multiplication with 0)
        feedforward_mult = np.array([[40]])

        outgoing_bp_error = nn.BackPropagation.compute_current_layer_error_deriv(
            incoming_bp_error, weights, feedforward_mult, nn.Sigmoid.derivative, with_bias=False)

        assert outgoing_bp_error == 0

if __name__ == '__main__':
    unittest.main()