import numpy as np
import urllib.request
import os.path

class Dataset:

    # TODO test
    def normalize(self, x, train_min=None, train_max=None):

        if train_min is None:
            train_min = np.min(x)
        if train_min > 0:
            x -= train_min

        if train_max is None:
            train_max = np.max(np.abs(x))
        return x / train_max, train_min, train_max

    # TODO test it
    @staticmethod
    def transform_y_from_label_values_to_label_indices(y, nr_labels):
        # for every y=i value, transform it into an array where v[i] = 1
        #assert np.min(y) == 0 and np.max(y) == nr_labels-1
        new_y_train = np.ndarray((y.shape[0], nr_labels))

        for i in range(nr_labels):
            new_y_train[:, i] = (y == i).astype(int)

        return new_y_train

    # TODO test it
    @staticmethod
    def get_columns_with_constant_values_for_all_entries(x_train):
        indexes=[]
        for i in range(x_train.shape[1]):
            # if all values in that column are equal to the first value in the column
            if np.count_nonzero(x_train[:, i] == x_train[0, i]) == x_train.shape[0]:
                indexes.append(i)
        return np.array(indexes)

    # TODO test it
    @staticmethod
    def subsampling_average(x_train):

        # 1st dimension is entry/row/image number
        # 2nd and 3rd is the Y and X of the image
        assert np.ndim(x_train) == 3

        result = np.zeros( (x_train.shape[0],
                            int(x_train.shape[1]/2 + x_train.shape[1]%2),
                              int(x_train.shape[2]/2 + x_train.shape[2]%2)))

        for image_i in range(result.shape[0]):
            for i in range(result.shape[1]):
                for j in range(result.shape[2]):
                    result[image_i, i, j] += x_train[image_i, i*2, j*2]
                    counter = 1

                    if i * 2 + 1 < x_train.shape[1]:
                        result[image_i, i, j] += x_train[image_i, i * 2 + 1, j * 2]
                        counter += 1

                    if j * 2 + 1 < x_train.shape[2]:
                        result[image_i, i, j] += x_train[image_i, i * 2, j * 2 + 1]
                        counter += 1

                    if i*2+1 < x_train.shape[1] and j*2+1 < x_train.shape[2]:
                        result[image_i, i, j] += x_train[image_i, i * 2 + 1, j * 2 + 1]
                        counter += 1

                    result[image_i, i, j] /= counter

        return result

    def deskew_image(self):
        """ From Lecun98: The deslanting computes the second moments of inertia of the pixels
        (counting a foreground pixel as 1 and a background pixel as 0 and shears the image by
        horizontally shifting the lines so that the principal axis is vertical.
        This version of the database will be referred to as the deslanted database"""


class MnistDataset(Dataset):
    def get_dataset(self, urlretrieve = urllib.request.urlretrieve):
        filename = 'mnist.npz'

        if not os.path.exists(filename):
            url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
            urlretrieve(url+filename, filename)

        with np.load(filename) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']

            return (x_train, y_train), (x_test, y_test)
