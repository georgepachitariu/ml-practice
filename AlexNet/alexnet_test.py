import tensorflow as tf
import unittest
import training
import numpy as np


# In this script I test that the code works as expected.
# Since I only have a small dataset, I don't need performance. So this code runs on CPU (it can also run on Github->Travis)
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')


# Create Tensor from unit-test image
def setup_dataset() -> tf.data.Dataset:
    ds = tf.data.Dataset.list_files('test_images/*')
    return ds.map(lambda t : {'image': t, 'label':'wolf'})
    return ds.map(process_path)

class TrainingTests(unittest.TestCase):

    def test_prepare(self):
        # assert that, afger prepare, the images have the right size
        # and same value range
        ins = setup_dataset()
        
        for d_in in ins:
            d_out = training._normalize(d_in)
            i = d_out['image'].numpy()
            assert i.shape == (training.IMG_SIZE, training.IMG_SIZE)
            assert np.all(-0.5 <= i) and np.all(i <= 0.5)

def test_print():
    list_ds = setup_dataset()
    for f in list_ds.take(1):
        print(f['image'].numpy())
        print(f['label'].numpy())

if __name__ == '__main__':
    unittest.main()