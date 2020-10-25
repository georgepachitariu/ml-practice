import numpy as np
from linearclassifier import LinearClassifier
import dataset.dataset as data
import unittest
import time


class LinearClassifierBenchmark(unittest.TestCase):

    def test_run(self):
        start = time.time()
        print("Linear Classifier Benchmark. Configuration: (784, 10). Test started.")

        dataset = data.MnistDataset()
        (x_train, y_train), (x_test, y_test) = dataset.get_dataset()

        x_train = x_train.reshape((60000, 784))
        x_train = dataset.normalize(x_train)
        y_train = dataset.transform_y_from_label_values_to_label_indices(y_train, nr_labels=10)

        classifier=LinearClassifier([x_train.shape[1], y_train.shape[1]], with_bias=True,
                                    regularization_value=0.01, random_seed=0)
        classifier.train(x_train, y_train, iterations=200, learning_rate=0.5,
                         debug_enabled=True, debug_intervales=30)

        x_test = x_test.reshape((10000, 784))
        x_test = dataset.normalize(x_test)
        y_test = dataset.transform_y_from_label_values_to_label_indices(y_test, nr_labels=10)

        predicted=classifier.predict(x_test)

        count, percentage = classifier.get_predicted_correctly(predicted, y_test)
        print("Linear Classifier Benchmark: Test set: " + str(count) + " predicted correctly. "
              "That is " + str(percentage) + "%")

        assert percentage > 87

        print("Linear Classifier Benchmark: Time taken: " + str(int(time.time() - start)) + "s.")
        #Linear Classifier Benchmark: Test set: 8855 predicted correctly. That is 88.55%


if __name__ == '__main__':
    unittest.main()

