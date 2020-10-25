import neuralnetwork as nn
import unittest
import numpy as np
import dataset.dataset as d

class NeuralNetworkExercises(unittest.TestCase):
    """
    Here there are exercises/tests that a network needs to be able to learn/train.
    (You can see them also as integration/scenario tests)
    """

    def test_function2_2layers_2features(self):
        # def test_function2_2layers(self):
        #   f is the xor function: f(x1, x2)= 1 when x1 != x2
        #                          f(x1, x2)= 0 when x1=x2
        x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_train = np.array([    0,      1,      1,      0])

        x_train, _, _ = d.Dataset().normalize(x_train)
        y_train = d.Dataset.transform_y_from_label_values_to_label_indices(y_train, 2)

        # In the most efficient configuration only 2+1(bias) nodes are needed in the hidden layer,
        # but in practice it doesn't converge. (I don't know why)
        classifier = nn.NeuralNetwork(layers_shape=[2, 5, 2], with_bias=True,
                                      regularization_value=0.0, random_seed=0)

        classifier.train(x_train, y_train, learning_rate=0.1, iterations=10000)

        # the network should be able predict all train cases correctly
        predicted=classifier.predict(x_train)
        _, percentage = classifier.get_predicted_correctly(predicted, y_train)
        assert percentage == 100


