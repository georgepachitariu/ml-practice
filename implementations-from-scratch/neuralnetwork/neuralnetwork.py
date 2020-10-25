import numpy as np
import warnings
import tools.tools as t


class Sigmoid:
    """Sigmoid is an activation function"""

    def compute(value):
        value[value < -10] = -10
        return np.power(1 + np.exp(-value), -1)

    def derivative(value):
        return Sigmoid.compute(value) * (1 - Sigmoid.compute(value))

class FeedForward:

    def multiply_weights_and_input(prev_activation_value, weights):
        # prev_activation_value is the input, it's also training data input (X) for the first layer

        # 1st dimension is entry/row number
        # 2nd dimension is layer node number (0 is the Bias node)
        assert np.ndim(prev_activation_value) == 2

        # 1st dimension is the next layer node number
        # 2nd dimension is current layer node number
        assert np.ndim(weights) == 2

        # the shape of assert np.ndim(activation_value) == 2
        result = np.dot(prev_activation_value, np.transpose(weights))
        # shape(result) is the same as the shape of activation_data,
        # where the number of nodes = next layer number of nodes
        return result

    def compute_activation_value(training_x, weights, activation_compute):
        weights_and_input_multiply = FeedForward.multiply_weights_and_input(training_x, weights)
        return weights_and_input_multiply, activation_compute(weights_and_input_multiply)

    # TODO Write test
    def run(layers_weights, x_train, with_bias):
        activation_values = []
        feedforward_mult = []
        for i in range(len(layers_weights)):
            # previus_layer_activation_values
            prev_layer_activ_val = x_train if i == 0 else activation_values[i - 1]

            if with_bias:
                prev_layer_activ_val = t.Tools.add_bias(prev_layer_activ_val)

            (weight_input_mult, activ_val) = FeedForward.compute_activation_value(
                prev_layer_activ_val, layers_weights[i], Sigmoid.compute)

            feedforward_mult.append(weight_input_mult)
            activation_values.append(activ_val)

        return feedforward_mult, activation_values


"""
In compute_last_layer_error you compute how bad is the network predicting by 
    comparing against training_y.
    
In compute_current_layer_error you compute how bad is the network predicting 
   the second to last layer by comparing agains training_y 
   BUT looking through the final set of weights.
"""
class BackPropagation:

    last_count_perc = 0
    def check_for_zeroes(error):
        perc = int(np.count_nonzero(np.abs(error) < 0.0001) / np.prod(error.shape) * 100)
        if perc > BackPropagation.last_count_perc:
            warnings.warn("Number of backpropagation errors with value zero increased to: " +
                          str(perc) + "% of total errors")
            BackPropagation.last_count=perc

    def compute_last_layer_error_deriv(activation_value, training_y, debug=False):
        """
        :param activation_value: This is the predicted value. It is the multiplication result
            of weights and input and after it goes through the activation function.
         Notice how we use the activation_value in the last layer but the feedforward_mult in
            the previous layers
        :param training_y: The training output value (the label). It is reshaped as an array where
            the value(label index)=1 and the rest are 0.
        :return: the backpropagation error
        """

        # 1st dimension is entry/row number
        # 2nd dimension is label node number
        assert np.ndim(activation_value) == np.ndim(training_y) == 2

        result = activation_value - training_y

        if debug:
            BackPropagation.check_for_zeroes(result)

        return result


    @staticmethod
    def compute_current_layer_error_deriv(error_previous_layer, layer_weights,
                                          feedforward_mult, activation_function_derivative,
                                          with_bias):
        """
        :param error_previous_layer: Backpropagation works backwards, from the final layer,
            until the first layer, so error_previous_layer is in this case "current_layer+1"

        :param layer_weights: weights of the current layer.

        :param feedforward_mult: this is the multiplication result of weights and input,
            before the activation function is applied to it.
            shape(feedforward_mult)=(n,1), where n is the number of nodes in the current layer

        :param activation_function_derivative: Sigmoid is one example of activation function.

        :return: the error.   shape(error) = (n,1) - where n is the number of nodes of the current layer.
            It's a sum over all the errors for each input record

        As example, when running for the penultimate layer, this is from which layer
         each variable comes from:
            1. error_previous_layer = last layer
            2. layer_weights = weights before last layer
            3. feedforward_mult = last layer
            4. activation_value = penultimate layer
        """

        # 1st dimension is entry/row number
        # 2nd dimension is layer node number (0 is the Bias node)
        assert np.ndim(error_previous_layer) == np.ndim(feedforward_mult) == 2

        # 1st dimension is layer node number
        # 2nd dimension is layer node number of the previous layer, going backwards (double negation)
        assert np.ndim(layer_weights) == 2

        # error derivative with respect to x, where x is the feedforward_mult
        error_deriv_x = activation_function_derivative(feedforward_mult) * error_previous_layer

        # np.ndim(bp_error) == 2 # the same as error_previous_layer
        # this is also error function derivative with respect to the activ. values of the penultimate layer
        error = np.dot(error_deriv_x, layer_weights)

        if with_bias:
            # We remove backpropagation error of the Bias node, because we cannot change it's input value
            error = error[:, 1:]

        BackPropagation.check_for_zeroes(error)

        return error

    def run(feedforward_mult, activation_values, y_train, layers_weights, with_bias, debug=False):
        # TODO: Backpropagation error of the last layer is computed differentely than
        #   the backpropagation of the non-last layer
        #  Test that "the behaviour is consistent": do the error values have the same sign (-/+) ?
        #                                         do the error values have the same range (interval) ?
        bp_error = []
        layer_error=BackPropagation.compute_last_layer_error_deriv(activation_values[-1], y_train, debug)

        bp_error.append(layer_error)

        for i in range(1, len(activation_values)):
            error=BackPropagation.compute_current_layer_error_deriv(error_previous_layer=bp_error[0],
                                                                    layer_weights=layers_weights[-i],
                                                                    feedforward_mult=feedforward_mult[-i],
                                                                    activation_function_derivative=Sigmoid.derivative,
                                                                    with_bias=with_bias)
            # in Backpropagation we insert the values backwards, at index 0, like in a stack
            bp_error.insert(0, error)

        return bp_error


class Cost:
    def get_regularization_term(reg_value, nr_examples, weights_list):
        """
        :param reg_value: this is the lambda
        """
        if reg_value == 0:
            return 0

        weights_l_squared_sum = [np.sum(np.power(el, 2)) for el in weights_list]
        weights_squared_sum = np.sum(weights_l_squared_sum)

        return reg_value / (2 * nr_examples) * weights_squared_sum

    def compute(predicted, y, weights_list, reg_value):

        # 1st dimension is entry/row number
        # 2nd dimension is label node number
        assert np.ndim(predicted) == np.ndim(y) == 2

        # each layer is an element in list
        len(weights_list) > 0
        for el in weights_list:
            # 1st dimension is entry/row number
            # 2nd dimension is label node number
            assert np.ndim(el) == 2

        nr_examples = np.shape(predicted)[0]

        predicted[predicted == 0] = 0.001
        predicted[predicted == 1] = 0.999

        return -(1.0 / nr_examples) * np.sum(y * np.log(predicted) + (1 - y) * np.log(1 - predicted)) + \
               Cost.get_regularization_term(reg_value, nr_examples, weights_list)

    def derivative(backpropagation_error, activation_values, layer_weights, reg_value):
        """The result is not computed using the derivative,
        but with another function that the derivative is equal to.

        In a NeuralNetwork with 1 layer Cost.compute will have the same shape as the final layer,
        but Cost.derivative will have the same shape as the current layer (multiplied for each row)

        As example, when running for the last layer, this is from which layer
         each variable comes from:
            1. backpropagation_error = last layer,
            2. activation_values = penultimate layer,
            3. layer_weights = last layer
        """

        # 1st dimension is entry/row number
        # 2nd dimension is layer node number (0 is the Bias node)
        assert np.ndim(backpropagation_error) == np.ndim(activation_values) == 2

        # 1st dimension is the next layer node number
        # 2nd dimension is current layer node number
        assert np.ndim(layer_weights) == 2

        nr_examples = np.shape(backpropagation_error)[0]

        # TODO I think that layer_weights should be squared
        result = (1.0 / nr_examples) * \
                 np.transpose(np.dot(np.transpose(activation_values), backpropagation_error)) + \
                 reg_value * layer_weights

        # It will have the same shape as the weights.
        assert np.shape(result) == np.shape(layer_weights)

        return result

    # TODO Test
    @staticmethod
    def derivative_all_layers(activation_values, train_x, bp_error, layers_weights, add_bias, regularization_value):
        cost_derivative=[]
        for i in range(len(activation_values)):
            activ_val = train_x if i==0 else activation_values[i-1]
            activ_val = t.Tools.add_bias(activ_val) if add_bias else activ_val

            deriv = Cost.derivative(bp_error[i], activ_val,
                                    layers_weights[i], regularization_value)
            cost_derivative.append(deriv)
        return cost_derivative


class NeuralNetwork:
    def __init__(self, layers_shape=None, layers_weights=None, with_bias=True,
                 regularization_value=0, random_seed=None):
        if layers_shape is not None:
            add_bias = 1 if with_bias else 0

            self.layers_weights = []
            for i in range(len(layers_shape) - 1):
                # 1st dimension is the next layer node number
                # 2nd dimension is current layer node number
                self.layers_weights.append(
                    NeuralNetwork.generate_random(layers_shape[i + 1],
                                                  layers_shape[i] + add_bias,
                                                  random_seed))
        else:
            self.layers_weights = layers_weights

        self.regularization_value = regularization_value
        self.with_bias = with_bias

    @staticmethod
    def generate_random(m, n, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        # the numbers are in the interval [-1,1)
        return np.random.random((m, n)) * 2 - 1

    @staticmethod
    def get_predicted_correctly(predicted, y):
        count = np.count_nonzero(np.argmax(predicted, axis=1) == np.argmax(y, axis=1))
        percentage = count / predicted.shape[0] * 100
        return count, percentage

    # TODO write small training cases with specified values and weights to see that it trains
    #           (like weights change in the right direction)
    def run_iteration(self, x_train, y_train, learning_rate=0.1, compute_cost=False, debug=False, i=-1):

        (feedforward_mult, activation_values) = FeedForward.run(self.layers_weights,
                                                                x_train, self.with_bias)

        cost = None
        if compute_cost:
            cost = Cost.compute(activation_values[-1], y_train,
                                 self.layers_weights, self.regularization_value)
            print("Cost: " + str(cost))

        if debug:
            print("Iteration: " + str(i))
            #how many were predicted correctly
            count, percentage = NeuralNetwork.get_predicted_correctly(activation_values[-1], y_train)
            print(str(count) + " predicted correctly. That is " + str(percentage) + "%")

            if percentage == 100.0:
                return 0

        bp_error = BackPropagation.run(feedforward_mult, activation_values, y_train,
                                       self.layers_weights, self.with_bias, debug)

        cost_derivative = Cost.derivative_all_layers(activation_values, x_train, bp_error,
                                                     self.layers_weights, self.with_bias, self.regularization_value)
        for i in range(0, len(self.layers_weights)):
            self.layers_weights[i] -= cost_derivative[i] * learning_rate

        # This return statement is added for tests, and not the algorithm itself
        return cost, bp_error, cost_derivative

    def train(self, x_train, y_train, learning_rate, iterations=1000, debug_enabled=False, debug_intervales=50):
        for i in range(iterations):
            debug = debug_enabled and i % debug_intervales == 0
            return_val = self.run_iteration(x_train, y_train, learning_rate, debug, debug, i)

            if return_val == 0:
                return

    def predict(self, x_test):
        (_, activation_values) = FeedForward.run(self.layers_weights, x_test, self.with_bias)
        return activation_values[-1]
