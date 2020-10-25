import numpy as np
from knearestneighbour import KNearestNeighbour
import dataset.dataset as dataset


if __name__ == '__main__':
    dataset = dataset.MnistDataset()
    (x_train, y_train), (x_test, y_test) = dataset.get_dataset()

    x_train = x_train.reshape((60000, 784))
    x_train = dataset.normalize(x_train)
    x_test = x_test.reshape((10000, 784))
    x_test = dataset.normalize(x_test)

    predicted = KNearestNeighbour().predict(x_train, y_train, x_test[:600])

    count = np.count_nonzero(predicted == y_test[:600])
    percentage = count / predicted.shape[0] * 100
    print(str(count) + " predicted correctly. That is " + str(percentage) + "%")

    # Benchmark with full training set but only 600 of the test rows (because on CPU it takes 0.9 sec. per row)
    # 578 predicted correctly. That is 96.33333333333334%