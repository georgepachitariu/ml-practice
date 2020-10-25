import numpy as np
import tools.tools as t
import time

class SupportVectorMachine():
    # SVM trains a line that separates positive examples from negative ones

    def __init__(self, number_input_features, number_labels, delta, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        # the numbers are in the interval [-1, 1)
        # "number_input_features + 1" because there is 1 for the bias
        self.weights = np.random.random((number_labels, number_input_features+1)) * 2 - 1

        self.delta = delta

    # TODO test it
    @staticmethod
    def apply_gaussian_kernel(input, gamma):
        """
        :param input: training data
        :param gamma: equals to 1 / (2 * sigma ^ 2). Recommended value = 1 / number of features:
        https://stats.stackexchange.com/questions/37681/use-of-the-gamma-parameter-with-support-vector-machines/37713#37713
        :return: the training data with the kernel applied
        """

        # 1st dimension is entry/row number
        # 2nd dimension is feature number
        # input.shape = (m, n)
        assert np.ndim(input) == 2

        number_elements = input.shape[1]
        new_input = np.zeros((input.shape[0], int((number_elements - 1) * number_elements / 2)), dtype=np.float16)

        # TODO you can also try with np.roll()
        count = 0
        for i in range(input.shape[1] - 1):
            for j in range(i + 1, input.shape[1]):
                new_input[:, count] = np.exp(-1 * np.power(input[:, i] - input[:, j], 2) * gamma)
                count += 1

        return new_input

    @staticmethod
    def compute_hinge_loss_term(input, weights, y, delta):
        # computes the term used in the multiclass hinge loss

        # 1st dimension is entry/row number
        # 2nd dimension is feature number
        # input.shape = (m, n)
        assert np.ndim(input) == 2

        # 1st dimension is weight-index, each output label has it own row of weights
        # 2nd dimension, J, is the index of the weight for input feature J
        # input.shape = (p, n)
        assert np.ndim(weights) == 2

        # 1st dimension is entry/row number
        # 2nd dimension has number of labels: has 1 on y[:, label] else 0
        # input.shape = (m, p)
        assert np.ndim(y) == 2

        # result.shape = (m, p)
        predicted = np.dot(input, np.transpose(weights))

        number_input_rows = input.shape[0]

        # We compute the difference, for the same input entry,
        # between the prediction for the correct label and the predictions for the other labels
        correct_label_prediction = predicted[y == 1].reshape(number_input_rows, 1)

        # Here there are 2 implicit broadcasts:
        #   1. product.shape = (m, p) & correct_label_for_row.shape=(m, 1)
        #   2. delta is a double
        diff = (predicted - correct_label_prediction) + delta

        return diff

    @staticmethod
    def compute_loss(hinge_loss_term):
        diff = hinge_loss_term
        diff[diff < 0] = 0 # max(0, diff)
        return np.nansum(diff)

    @staticmethod
    def compute_gradient_analitically(input, weights, y, hinge_loss_term, regularization_value):

        # The row is the input entry for which the loss was computed for;
        # The column is the label weight vector, for which the loss was computed for;
        # The loss is a scalar;
        assert np.ndim(hinge_loss_term) == 2

        # As you see in this methof, if hinge_loss_term < 0, the gradient is not affected anymore by the
        # result of the "hinge_loss_term". "It doesn'T try to strecth/separate the 2 class labels"
        hinge_loss_term[hinge_loss_term < 0] = 0
        # If hinge_loss_term[i,j] > 0, it meas that for row i and label j, the label is incorrect and
        # it had a higher predicted value than the correct label. These are the "support vectors".
        # Support vectors are a few input entries that are used to train the model (separate the classes).
        hinge_loss_term[hinge_loss_term > 0] = 1

        gradient = np.zeros_like(weights)

        normalization_counter = 0
        for entry_row_i in range(input.shape[0]):
            correct_label = -1

            count = 0
            for label_col_j in range(hinge_loss_term.shape[1]):  # number iterations = number labels
                # if the label is not the correct one:
                if y[entry_row_i, label_col_j] != 1:
                    if hinge_loss_term[entry_row_i, label_col_j] == 1: # it can be 1 or 0
                        count += 1  # this is a sum of ones; max = count(labels)-1
                        # ib: for the incorrect label, we add single losses multiplied by the input entry:
                        gradient[label_col_j] += 1 * input[entry_row_i]
                else:
                    correct_label = label_col_j
            if count > 0:
                # 1a: for the correct label, we make the sum of all the incorrect label an multiply if with thw input row
                gradient[correct_label] -= count * input[entry_row_i]
                normalization_counter += 1

        if normalization_counter > 0:
            gradient /= normalization_counter  # normalization

        # TODO test with weights squared
        gradient += regularization_value * weights

        return gradient

    def train(self, x_train, y_train, learning_rate=0.1, regularization_value=0.01, iterations=10000, run_on_gpu=False):

        x_train = t.Tools.add_bias(x_train)

        for i in range(iterations):
            hinge_loss_term = SupportVectorMachine.compute_hinge_loss_term(
                x_train, self.weights, y_train, self.delta)

            loss = SupportVectorMachine.compute_loss(np.copy(hinge_loss_term))

            if run_on_gpu:
                import supportvectormachine_gpu as svm_gpu
                gradient = svm_gpu.SupportVectorMachineGPU.compute_gradient_analitically(
                        x_train, self.weights, y_train, np.copy(hinge_loss_term), regularization_value)
            else:
                gradient = SupportVectorMachine.compute_gradient_analitically(
                        x_train, self.weights, y_train, np.copy(hinge_loss_term), regularization_value)

            if i % 20 == 0:
                print("Loss: " + str(loss))
                if loss == 0.0:
                    return

                _, train_perc = SupportVectorMachine.get_predicted_correctly(
                    self.predict(x_train, has_bias_already=True), y_train)
                print('Iteration ' + str(i) + '. Train set: ' + str(train_perc) + '% predicted correctly.')

            self.weights -= learning_rate * gradient

    def predict(self, x_test, has_bias_already=False):
        if not has_bias_already:
            x_test = t.Tools.add_bias(x_test)
        return np.dot(x_test, np.transpose(self.weights))

    @staticmethod
    def get_predicted_correctly(predicted, y):
        count = np.count_nonzero(np.argmax(predicted, axis=1) == np.argmax(y, axis=1))
        percentage = count / predicted.shape[0] * 100
        return count, percentage
