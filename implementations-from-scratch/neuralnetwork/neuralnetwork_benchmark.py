import numpy as np
from neuralnetwork import NeuralNetwork
import dataset.dataset as data
import unittest
import time


class LinearClassifierBenchmark(unittest.TestCase):

    def test_run(self):
        start = time.time()
        print("Neural Network Benchmark. Configuration: (784, 300, 10). Test started.")

        # TODO: I should use "mean square error" <- Is this implemented already?

        dataset = data.MnistDataset()
        (x_train, y_train), (x_test, y_test) = dataset.get_dataset()

        x_train = x_train.reshape((60000, 784))
        x_train = dataset.normalize(x_train)
        y_train = dataset.transform_y_from_label_values_to_label_indices(y_train, nr_labels=10)

        hidden_layer = 300
        classifier=NeuralNetwork(layers_shape=[x_train.shape[1], hidden_layer, y_train.shape[1]], with_bias=True,
                                 regularization_value=0.001, random_seed=0)
        classifier.train(x_train, y_train, iterations=2, learning_rate=0.3,
                         debug_enabled=True, debug_intervales=30)

        x_test = x_test.reshape((10000, 784))
        x_test = dataset.normalize(x_test)
        y_test = dataset.transform_y_from_label_values_to_label_indices(y_test, nr_labels=10)

        predicted=classifier.predict(x_test)

        count, percentage = classifier.get_predicted_correctly(predicted, y_test)
        print("Neural Network Benchmark: Test set: " + str(count) + " predicted correctly. " +
              "That is " + str(percentage) + "%")

        print("Neural Network Benchmark: Time taken: " + str(int(time.time() - start)) + "s.")
        # Neural Network Benchmark: Test set: 9137 predicted correctly. That is 91.36999999999999%

        # TODO So far the record is 91.3%. It's not done, the Lecun target is 95.3%.
        #  But I cannot iterate fast and try things if trying out one time takes 1 hour
        #  Reimplement it to work on GPU
         # Full train set, iterations=1000, learning_rate=0.5,
        # Iteration: 990
        # 54647 predicted correctly. That is 91.07833333333333%
        # Neural Network Benchmark: Test set: 9137 predicted correctly. That is 91.36999999999999%
        # Neural Network Benchmark: Time taken: 4166s.

"""
Last output: 
1. Cost continues to decrease, 
2. #predicted increases
3. Difference in predicted percentage between train set and test set is 0.

Full train set, iterations=1000, learning_rate=0.5,
Iteration: 990
54647 predicted correctly. That is 91.07833333333333%
Neural Network Benchmark: Test set: 9137 predicted correctly. That is 91.36999999999999%
Neural Network Benchmark: Time taken: 4166s.
"""


if __name__ == '__main__':
    unittest.main()



