import tensorflow as tf
import tensorflow_datasets as tfds
import unittest
import numpy as np
from contextlib import contextmanager
import alexnet


# In this script I test that the code works as expected.
# Since I only have a small dataset, I don't need performance. So this code runs on CPU (it can also run on Github->Travis)
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')


def create_square(size = 224):
    square = np.zeros(shape=(size, size, 3), dtype=np.uint8)
    for i in range(int(size * 1/4), int(size * 3/4)):
        for j in range(int(size * 1/4), int(size * 3/4)):
            square[i, j, :] = 255
    return square

def create_circle(size = 224):
    # circle formula: (x - a)**2 + (y - b)**2 = r**2
    circle = np.zeros(shape=(size, size, 3), dtype=np.uint8)
    center = size/2
    r = size/2
    for i in range(circle.shape[0]):
        for j in range(circle.shape[1]):
            if (i - center)**2 + (j - center)**2 < r**2 :
                circle[i, j, :] = 255
    return circle

def as_dataset(self, *args, **kwargs):
    return tf.data.Dataset.from_tensor_slices(
        { 'image': [create_square(), create_circle()],
         'label': [0, 1]}
    ) 


class TrainingTests(unittest.TestCase):

    @contextmanager
    def test_with_basic_dataset(self):
        # The 'with ... :' makes tfds.load() to load my above mock dataset instead of imagenet2012.
        # Both train and validation split will contain all the elements.
        with tfds.testing.mock_data(as_dataset_fn=as_dataset):
            history = alexnet.run(
                #len(train_data) >= steps_per_epoch * epochs * batch_size
                number_categories=2, dataset_repeat = 750, steps_per_epoch=5, epochs=30, batch_size=10, sample_fraction=1,
                # The learning rate needs to be very small, I think it's because of the small number of training data
                learning_rate=0.00001
                )
            print()

            assert history.history['accuracy'][-1] == 1.0 


if __name__ == '__main__':
    unittest.main()
