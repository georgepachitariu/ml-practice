import numpy as np
import dataset.dataset as data
import unittest
import time
import supportvectormachine as svm


class SupportVectorMachineBenchmark(unittest.TestCase):

    def test_run(self):
        start = time.time()
        print("Support Vector Machine Benchmark started.")

        dataset = data.MnistDataset()
        (x_train, y_train), (x_test, y_test) = dataset.get_dataset()

        # Get a subsample for each image:
        x_train = data.Dataset.subsampling_average(x_train.reshape((-1, 28, 28))).reshape((-1, 196))
        x_test = data.Dataset.subsampling_average(x_test.reshape((-1, 28, 28))).reshape((-1, 196))

        # Normalize
        x_train, train_min, train_max = dataset.normalize(x_train)
        x_test, _, _ = dataset.normalize(x_test, train_min=train_min, train_max=train_max)

        # Apply Gaussian kernel:
        x_train = svm.SupportVectorMachine.apply_gaussian_kernel(x_train, gamma=1)
        x_test = svm.SupportVectorMachine.apply_gaussian_kernel(x_test, gamma=1)

        y_train = dataset.transform_y_from_label_values_to_label_indices(y_train, nr_labels=10)
        y_test = dataset.transform_y_from_label_values_to_label_indices(y_test, nr_labels=10)

        classifier = svm.SupportVectorMachine(
            number_input_features=19110, number_labels=10, delta=1, random_seed=0)

        classifier.train(x_train, y_train, learning_rate=1, regularization_value=0.0001,
                         iterations=3000, run_on_gpu=True)

        predicted = classifier.predict(x_test)

        count, percentage = classifier.get_predicted_correctly(predicted, y_test)
        print("Support Vector Machine Benchmark: Test set: " + str(count) + " predicted correctly. " +
              "That is " + str(percentage) + "%")

        print("Support Vector Machine Benchmark: Time taken: " + str(int(time.time() - start)) + "s.")

# I got the result:
#   Support Vector Machine Benchmark: Test set: 9164 predicted correctly. That is 91.64%
#   Support Vector Machine Benchmark: Time taken: 46773s.
#
# I used the entire training dataset, but I did subsampling of the images. Gaussian kernel increases
# the number of feature from N to N * (N-1), where N is 784 in the original and 196 with subsampling.
#
# My result for SVM with Gaussian kernel is less than the one in the LeCun paper. Their error is 1.4% and mine is 8.4%.
# One cause for the difference is because I did subsampling.
# Another cause, is that I stopped the training after 3000 iterations (12 hours).
# The accuracy of the model was still increasing when I stopped it.
#
# TODO Future work: Run it with no subsampling.
# The only problem with this is that it needs a high computational power (or it takes weeks).
# The expensive part (Gradient computing) is already implemented in Cuda, for GPU.

if __name__ == '__main__':
    unittest.main()



