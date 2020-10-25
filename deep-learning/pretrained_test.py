from data import Imagenet2012
import gpu
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from tensorflow.keras.layers import Conv2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, \
    Flatten, Dropout, Dense, ReLU, Lambda, Softmax
from tensorflow.dtypes import cast
import os, sys, shutil

from resnet import Preprocessing
import tensorflow_hub as hub

class PretrainedResnet(tf.keras.Model):

    def __init__(self):
        super(PretrainedResnet, self).__init__(name='')

        self.model= tf.keras.Sequential([
            hub.KerasLayer(
                "https://tfhub.dev/tensorflow/resnet_50/classification/1",
                #"https://tfhub.dev/google/imagenet/resnet_v1_50/classification/4",
               trainable=False)
            ])

    def call(self, input_tensor, training=False):
        return self.model(input_tensor, training=training)
    
    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        images_crops, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        total_number_crops = tf.shape(images_crops)[0]
        distinct_images = total_number_crops / 10 # there are 10 crops per image
        # Segments will look like this: [0, 0, ..., 1, 1, ...].
        segments = tf.repeat(tf.range(0, distinct_images, 1, dtype=tf.int32), repeats=10)

        y_pred_crops=self.model(images_crops, training=False)
        # I segment to get the mean score per 10 crops for each image. Check the testing preprocessing as well.
        y_pred = tf.math.segment_mean(y_pred_crops, segments)
        y = tf.math.segment_max(y, segments) # I segment y based on same rules to have the same final shape

        # Updates stateful loss metrics.
        self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def compile(self, learning_rate=0.1):
        super().compile(
            # Also check the ReduceLROnPlateau training callback lower in script.
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
            # "categorical_crossentropy": uses a one-hot array to calculate the probability,
            # "sparse_categorical_crossentropy": uses a category index
            # source: https://stackoverflow.com/questions/58565394/what-is-the-difference-between-sparse-categorical-crossentropy-and-categorical-c
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)])


def main():
    gpu.configure_gpu()
    r = PretrainedResnet()
    r.compile()
    r.build(input_shape=(None, 224, 224, 3))
    
    total_train_data_size = 1281167
    total_validation_data_size = 50000

    train_data, validation_data = Imagenet2012.load()
    nr_images = 32*1000
    train_data = train_data.take(nr_images)
    validation_data = validation_data.take(total_validation_data_size)

    batch_size=32
    train_augmented_gen = Preprocessing.create_generator(train_data, for_training=True, batch_size=batch_size)
    validation_gen = Preprocessing.create_generator(validation_data, for_training=False, 
                    batch_size=None # batch_size is treated differently during validations
                    )

    auto=tf.data.experimental.AUTOTUNE
    train_augmented_gen = train_augmented_gen.map(lambda im, l: (im+0.5, l), num_parallel_calls=auto)
    validation_gen = validation_gen.map(lambda im, l: (im+0.5, l), num_parallel_calls=auto)


    log_dir = './logs/fit/jupyterhub_version'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=2)

    history = r.fit(x=train_augmented_gen,
                         validation_data=validation_gen,
                         initial_epoch=0, 
                         epochs=1,                   
                         # steps_per_epoch = total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch
                         steps_per_epoch=nr_images/batch_size,
                         callbacks=[tensorboard_callback]
                         ) 
    
if __name__ == '__main__':
    main()      

