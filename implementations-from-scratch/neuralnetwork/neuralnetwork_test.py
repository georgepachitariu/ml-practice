import neuralnetwork as nn
import unittest
import numpy as np
import warnings



class SigmoidTests(unittest.TestCase):
    def test_sigmoid(self):
        assert 0 < nn.Sigmoid.compute(np.array([-10])) < 0.0001
        assert nn.Sigmoid.compute(np.array([0])) == 0.5
        assert 0.9999 < nn.Sigmoid.compute(np.array([10])) < 1

    def test_sigmoid_derivative(self):
        assert 0 < nn.Sigmoid.derivative(np.array([-10])) < 0.0001
        assert nn.Sigmoid.derivative(np.array([0])) == 0.25
        assert 0 < nn.Sigmoid.derivative(np.array([10])) < 0.0001


class FeedForward(unittest.TestCase):
    def test_feedforward_2nodes_bias_nextlayer_2nodes_4records(self):
        x = np.array([[1,  2, 2],
                      [1,  0, 1],
                      [1, -1, 0],
                      [1,  1, 0]])
        weights = np.array([[-1.0, 1.0, 0],
                            [-0.1, 0.1, 0]])

        result = nn.FeedForward.multiply_weights_and_input(x, weights)

        assert np.array_equal(result, np.array([[1,   0.1],
                                                [-1, -0.1],
                                                [-2, -0.2],
                                                [0,    -0]]))


class CostTests(unittest.TestCase):
    def test_compute_regularization_term_2layers_weights(self):
        reg_value = 0.5
        nr_examples = 2
        weights = [np.array([[0.1, 0.2], [0.1, 0.2]]), np.array([[0.1, 0.2]])]
        result = nn.Cost.get_regularization_term(reg_value, nr_examples, weights)
        assert abs(result - 0.01875) < 0.0001

    def test_compute_cost_correct_1(self):
        predicted = np.array([[0.99999]])
        y = np.array([[1]])
        assert nn.Cost.compute(predicted, y, [], 0) < 0.0001

    def test_compute_cost_correct_2(self):
        predicted = np.array([[0.00001]])
        y = np.array([[0]])
        assert nn.Cost.compute(predicted, y, [], 0) < 0.0001

    def test_compute_cost_incorrect_1(self):
        predicted = np.array([[0.00001]])
        y = np.array([[1]])
        assert nn.Cost.compute(predicted, y, [], 0) > 1

    def test_compute_cost_incorrect_2(self):
        predicted = np.array([[0.99999]])
        y = np.array([[0]])
        assert nn.Cost.compute(predicted, y, [], 0) > 1

    def test_compute_cost_2records_2nodes_and_regularization(self):
        predicted = np.array([[0.99999], [0.00001]])
        y = np.array([[0],[0]])
        weights = [np.array([[0.1, 0.2], [0.1, 0.2]])]

        # -1/2 * (1*ln(1-0.99999)+1*ln(1-0.00001)) +
        #  0.5 / (2*2)*(0.01*2+0.04*2) = 5.7689

        result = nn.Cost.compute(predicted, y, weights, 0.5)
        assert 0 < result - 5.7689 < 0.0001

    def test_cost_compute_derivative_3nodes_4examples(self):
        # next layer has 2 nodes
        backpropagation_error=np.array([[0.5, 0.2],
                                        [0.5, 0.2],
                                        [0.5, 0.2],
                                        [0.5, 0.2]])
        activation_values=np.array([[0.2, 0.2, 0.2],
                                    [0.3, 0.3, 0.3],
                                    [0.4, 0.4, 0.4],
                                    [0.5, 0.5, 0.5]])
        reg_value=0.2
        layer_weights=np.array([[0.5, 0.5, 0.5],
                                [0.1, 0.1, 0.1]])

        #       (0.5, 0.5, 0.5, 0.5)   (0.2, 0.2, 0.2)   (0.175, 0.175, 0.175)
        # 1/4 * (0.2, 0.2, 0.2, 0.2) * (0.3, 0.3, 0.3) = (0.07,  0.07,  0.07 )
        #                              (0.4, 0.4, 0.4)
        #                              (0.5, 0.5, 0.5)

        # (0.175, 0.175, 0.175)         (0.5, 0.5, 0.5)   (0.275, 0.275, 0.275)
        # (0.07,  0.07,  0.07 ) + 0.2 * (0.1, 0.1, 0.1) = (0.09,  0.09,  0.09 )

        result = nn.Cost.derivative(backpropagation_error, activation_values,
                                    layer_weights, reg_value)
        assert np.allclose(result, np.array([[0.275, 0.275, 0.275],
                                             [0.09,  0.09,  0.09]]))


class BackpropagationTests(unittest.TestCase):
    def test_compute_error_final_layer_3labels_2records(self):
        feedforward_mult = np.array([[0.1, 0.1, 0.9], [0.4, 0.4, 0.5]])
        training_y = np.array([[0, 0, 1], [0, 1, 0]])

        result=nn.BackPropagation.compute_last_layer_error_deriv(feedforward_mult, training_y)
        result=np.around(result, decimals=1)

        assert np.array_equal(result, np.array([[0.1, 0.1, -0.1], [0.4, -0.6, 0.5]]))

    def test_compute_error_middle_layer_3nodes_bias_2records(self):
        # previous layer, in backpropagation, has 3 nodes

        error_previous_layer = np.array([[0.5, 0.5, 0.5], [0.1, 0.4, -0.3]])
        layer_weights = np.array([[1, 2, 3, 4],
                                  [1, 1, 1, 1],
                                  [1, 1, 1, 1]])

        feedforward_mult = np.array([[1, 2, 3], [0.1, 0.2, 0.3]])

        #                   feedforw      prev_err
        #  (1, 0.1)      (0.5, 0.1)      (0.5,  0.01)
        #  (2, 0.2)  *   (0.5, 0.4)  =   (1,    0.08)
        #  (3, 0.3)      (0.5,-0.3)      (1.5, -0.09)

        #  (0.5,  1,     1.5 )      (1, 2, 3, 4)    (3, 3.5,  4,    4.5 )
        #  (0.01, 0.08, -0.09)  *   (1, 1, 1, 1) =  (0, 0.01, 0.02, 0.03)
        #                           (1, 1, 1, 1)

        result=nn.BackPropagation.compute_current_layer_error_deriv(error_previous_layer, layer_weights,
                                                                    feedforward_mult, lambda i: i, with_bias=True)
        assert np.allclose(result, np.array([[3.5,  4,    4.5],
                                             [0.01, 0.02, 0.03]]))

    def test_warning_0error(self):
        feedforward_mult = np.array([[1,0]])
        training_y = np.array([[1,1]])

        with warnings.catch_warnings(record=True) as w:
            nn.BackPropagation.compute_last_layer_error_deriv(feedforward_mult, training_y, debug=True)
            assert str(w[0].message) == 'Number of backpropagation errors ' \
                                        'with value zero increased to: 50% of total errors'

class NeuralNetworkTests(unittest.TestCase):

    def test_basic(self):
        net=nn.NeuralNetwork(layers_shape=(2,3), with_bias=False)

        assert len(net.layers_weights) == 1
        assert net.layers_weights[0].shape == (3,2)
        assert np.all(-1 <= net.layers_weights[0]) and np.all(net.layers_weights[0] < 1)

    def test_basic_weights(self, ):
        net = nn.NeuralNetwork(layers_weights=[np.array([1])], regularization_value=5)

        assert net.layers_weights == [np.array([1])]
        assert net.regularization_value == 5

    def test_print_perc_predicted_correctly(self):
        predicted=np.array([[0.6, 0.2],
                            [0.2, 0.9],
                            [0.9, 0.99],
                            [0.8, 0.2]])

        y = np.array([[1, 0],
                      [1, 0],
                      [1, 0],
                      [0, 1]])

        count, percentage = nn.NeuralNetwork().get_predicted_correctly(predicted, y)
        assert count == 1
        assert percentage == 25



if __name__ == '__main__':
    unittest.main()
