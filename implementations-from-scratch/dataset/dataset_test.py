from dataset import Dataset, MnistDataset
import unittest
import os.path
from unittest.mock import MagicMock
import numpy as np


class DatasetTests(unittest.TestCase):
    def test_file_doesnt_exist(self):
        filename = 'mnist.npz'
        if os.path.exists(filename):
            os.remove(filename)

        (x_train, y_train), (x_test, y_test) = MnistDataset().get_dataset()
        assert x_train.shape == (60000, 28, 28)
        assert y_train.shape == (60000, )
        assert x_test.shape == (10000, 28, 28)
        assert y_test.shape == (10000, )

    def test_file_exists(self):
        # Above I tested that if the file doesn't exist it will download it
        # Here I test that if the file was already downloaded, it will not download it again

        # run it once to make sure the file is downloaded
        MnistDataset().get_dataset()

        urlretrieve = MagicMock()
        MnistDataset().get_dataset(urlretrieve)

        urlretrieve.method.assert_not_called()

    def test_subsampling_average(self):

        input1 = np.array([[[1]]])
        np.all(input1 == Dataset.subsampling_average(input1))

        input2 = np.array([[[1, 2]]])
        np.all(np.array([[[1.5]]]) == Dataset.subsampling_average(input2))

        input3 = np.array([[[1], [3]]])
        np.all(np.array([[2]]) == Dataset.subsampling_average(input3))

        input4 = np.array([[[1, 2], [3, 4], [5, 7]]])
        np.all(np.array([[[2.5], [6]]]) == Dataset.subsampling_average(input4))

        input5 = np.array([[[1]], [[1]]]) # 2 images
        np.all(input5 == Dataset.subsampling_average(input5))
