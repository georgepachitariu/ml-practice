import numpy as np

class KNearestNeighbour():
    def computeEuclideanDistance(self, a, b):
        return np.sqrt(np.sum(np.power(a-b, 2), axis=1))

    def getEuclideandistanceBetweenEachTrainAndTestRow(self, train, test):
        """
        For each test row we compute the euclidean distance to each train row
        :param train: the train set
        :param test: the test set
        :return: result, where number of rows = number of rows in test
                and number of columns = number of rows in train
        """
        result = np.empty((test.shape[0], train.shape[0]))

        for i in range(test.shape[0]):
            # here I'm doing implicit broadcasting: test[i] is a row but train is a matrix
            result[i] = self.computeEuclideanDistance(test[i], train)

            if i > 0:
                percentage_done = i/test.shape[0]*100
                if percentage_done % 5 == 0:
                    print("KNearestNeighbour: Evaluated " + str(i) + " test entries. That is " + str(percentage_done))

        return result

    def predict(self, train_x, train_y, test_x):
        """
        :param train_x: features of the train set, one row is one example
        :param train_y: label of the train set, one row is one example
        :param test_x: features of the test set, on row is one example
        :return: test_predicted: the predicted labels for each of the test set rows
        """

        m = self.getEuclideandistanceBetweenEachTrainAndTestRow(train_x, test_x)

        # get minimum per column
        closest_train_X_rows = np.argmin(m, axis=1)

        test_predicted = np.empty(test_x.shape[0])

        for i in range(test_x.shape[0]):
            test_predicted[i]=train_y[closest_train_X_rows[i]]

        return test_predicted



