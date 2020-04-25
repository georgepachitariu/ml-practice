import tensorflow as tf


# In this script I test that the code works as expected.
# Since I only have a small dataset, I don't need performance. So this code runs on CPU (it can also run on Github->Travis)
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

# Create Tensor from unit-test image
def setup_dataset() -> tf.data.Dataset:
    return tf.data.Dataset.list_files('test_images/*')

print(setup_dataset())