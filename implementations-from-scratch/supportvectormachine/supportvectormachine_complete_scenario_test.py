import dataset.dataset as d
import numpy as np
import unittest
import supportvectormachine as svm


class SupportVectorMachineExercises(unittest.TestCase):
    """
    Here there are exercises/tests that a SVM model needs to be able to learn/train.
    (You can see them also as integration/scenario tests)
    """

    def test_svm_separates_classes_through_middle(self):
        # SVM is known that it doesn't stop converging when it gets a separation of the classes.
        # It continues to converge until it creates a large margin between the classes.
        #
        # In this method I train SVM with 4 points, and then
        # I test with 4 points which are much closer to the middle than the train points.
        # Predicting them correctly means that SVM chose the limit close to the middle
        # (close to maximum margin on both sides)

        x_train = np.array([[2, 0], [4, 0], [0, 2], [0, 4]])
        y_train = np.array([0, 1, 0, 1])

        x_train, train_min, train_max = d.Dataset().normalize(x_train, 5)
        y_train = d.Dataset.transform_y_from_label_values_to_label_indices(y_train, 2)

        classifier = svm.SupportVectorMachine(
            number_input_features=2, number_labels=2, delta=5, random_seed=0)

        classifier.train(x_train, y_train, learning_rate=0.1, regularization_value=0.0, iterations=10000)

        x_test = np.array([[3.01, 0], [2.99, 0], [0, 2.99], [0, 3.01]])
        y_test = np.array([1, 0, 0, 1])

        x_test, _, _ = d.Dataset().normalize(x_test, train_min, train_max)
        y_test = d.Dataset.transform_y_from_label_values_to_label_indices(y_test, 2)

        predicted = classifier.predict(x_test)
        _, percentage = classifier.get_predicted_correctly(predicted, y_test)
        assert percentage == 100

    def test_svm_gaussian(self):
        ## same input set, but I apply gaussian kernel on it

        x_train = np.array([[2, 0], [4, 0], [0, 2], [0, 4]])
        y_train = np.array([0, 1, 0, 1])

        x_train, train_min, train_max = d.Dataset().normalize(x_train, 5)
        x_train = svm.SupportVectorMachine.apply_gaussian_kernel(x_train, gamma=1)

        y_train = d.Dataset.transform_y_from_label_values_to_label_indices(y_train, 2)

        classifier = svm.SupportVectorMachine(
            number_input_features=1, number_labels=2, delta=5, random_seed=0)

        classifier.train(x_train, y_train, learning_rate=1, regularization_value=0.0, iterations=10000)

        x_test = np.array([[2, 0], [4, 0], [0, 2], [0, 4]])
        y_test = np.array([0, 1, 0, 1])
        #x_test = np.array([[3.05, 0], [2.04, 0], [0, 2.04], [0, 3.05]])
        #y_test = np.array([1, 0, 0, 1])

        x_test, _, _ = d.Dataset().normalize(x_test, train_min, train_max)
        x_test = svm.SupportVectorMachine.apply_gaussian_kernel(x_test, gamma=1)

        y_test = d.Dataset.transform_y_from_label_values_to_label_indices(y_test, 2)

        predicted = classifier.predict(x_test)
        _, percentage = classifier.get_predicted_correctly(predicted, y_test)
        assert percentage == 100



if __name__ == '__main__':
    unittest.main()