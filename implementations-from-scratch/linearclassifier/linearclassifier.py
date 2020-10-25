import numpy as np
from neuralnetwork.neuralnetwork import NeuralNetwork

class LinearClassifier(NeuralNetwork):
    """
    A linear classifier is a neural network with 1 layer
    """
    def __init__(self,
                 layers_shape: int = None,
                 layers_weights=None,
                 with_bias=True,
                 regularization_value=0,
                 random_seed=None):

        super().__init__(layers_shape, layers_weights=layers_weights, with_bias=with_bias,
                         regularization_value=regularization_value, random_seed=random_seed)

