import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras


class Data:
    @staticmethod
    def load() -> (tf.data.Dataset, tf.data.Dataset):
        train_ds, validation_ds = tfds.load(name="imagenet2012", split=['train', 'validation'],
                                            data_dir='/home/gpachitariu/HDD/data')
        
        Data.test_assumptions_of_the_input(train_ds)

        return train_ds, validation_ds
    
    # The ML algorithm has a few assumptions of the input. We test the assumptions on the first example.
    # If the input data doesn't follow the assumptions training, testing & predicting will fail because 
    # the algorithm is "calibrated" the wrong way.
    @staticmethod
    def test_assumptions_of_the_input(train_ds):
        for d in train_ds.take(1):
            image = d['image'].numpy()
            # The image has 3 dimensions (height, width, color_channels). Also color_channels=3
            assert len(image.shape) == 3 and image.shape[2] == 3
            # The range of values for pixels are [0, 255]
            assert image.min() == 0
            assert image.max() == 255



class Preprocessing:
    # Notes
    # Resize: In the paper for images they kept the ratio, in mine the images were made square

    IMG_SIZE = 448  # TODO What should the Image size be?
    BATCH_SIZE = 128  # TODO is batch size correct?
    BUFFER_SIZE = 1000
    #train_image_count = tf.data.experimental.cardinality(train_ds).numpy()
    #STEPS_PER_EPOCH = np.ceil(train_image_count / BATCH_SIZE)

    @staticmethod
    def _split_dict(t: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        return t['image'], t['label']

    # TODO It would be nice to have a peek at the first image
    #    to see that the values are between -0.5 and 1
    # TODO is the type of the input tf.Tensor?
    @staticmethod
    def _normalize(image: tf.Tensor, label: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        # the values are between 0 and 255. We make them between -0.5 and 0.5
        image = (image / 255) - 0.5
        image = tf.image.resize(image, (Preprocessing.IMG_SIZE, Preprocessing.IMG_SIZE))
        return image, label

    @staticmethod
    def _augment(image: tf.Tensor, label: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        # These 4 (rotation, brightness, contrast, flip)
        # are added by me as a helper to get to 70% accuracy, not part of the paper
        # TODO img = tf.keras.preprocessing.image.random_rotation(rg=45, fill_mode='constant', cval=0)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

        crop_size = (tf.random.uniform([1])[0] * 0.25 + 0.75) * Preprocessing.IMG_SIZE
        # zoom in & out. max(zoom_out)=original size
        # image = tf.image.random_crop(image, size = (crop_size, crop_size))
        image = tf.image.resize(image, size=tf.constant((224, 224)))

        image = tf.image.random_flip_left_right(image)
        return image, label


    @staticmethod
    def create_generator(ds, for_training):
        auto=tf.data.experimental.AUTOTUNE

        ds = ds.map(Preprocessing._split_dict, num_parallel_calls=auto)
        ds = ds.map(Preprocessing._normalize, num_parallel_calls=auto)
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=Preprocessing.BUFFER_SIZE)

        if for_training:
            ds = ds.repeat(count=100)
            ds = ds.map(Preprocessing._augment, num_parallel_calls=auto)
        ds = ds.batch(Preprocessing.BATCH_SIZE)

        # dataset fetches batches in the background while the model is training.
        ds = ds.prefetch(buffer_size=auto)

        return ds




class Model:
    #TODO compute LEN_CLASS_NAMES
    LEN_CLASS_NAMES=1


    @staticmethod
    def build():
        # TODO At test time, the network makes a prediction by extracting five 224 × 224 patches
        # (the four corner patches and the center patch) as well as their horizontal reflections (hence ten patches in all),
        # and averaging the predictions made by the network’s softmax layer on the ten patches

        def normalized_init(number_inputs):
            return tf.compat.v1.keras.initializers.RandomNormal(mean=0.0, stddev=1 / number_inputs)

        k_init = tf.compat.v1.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        point_one = tf.compat.v2.constant_initializer(value=0.01)

        model = keras.Sequential([
            keras.layers.Conv2D(96, (11, 11),  strides=4, activation='relu', #TODO put it back: input_shape=(224, 224, 3),
                                bias_initializer='zeros', kernel_initializer=normalized_init(121)),
            keras.layers.MaxPooling2D(pool_size=3, strides=2),

            keras.layers.Conv2D(256, (5, 5), activation='relu', bias_initializer=point_one,
                                kernel_initializer=normalized_init(25)),
            keras.layers.MaxPooling2D(pool_size=3, strides=2),

            keras.layers.Conv2D(384, (3, 3), activation='relu', bias_initializer=point_one,
                                kernel_initializer=normalized_init(9)),

            # 4th conv. layer
            keras.layers.Conv2D(384, (3, 3), activation='relu', bias_initializer=point_one,
                                kernel_initializer=normalized_init(9)),

            # TODO: How does the Convolutional layer work?
            # 1. For every convolution from every filter, it connects with a (3,3) window to all 384 previous filters?
            # 2. Should it be 9 ? or 9*384?
            keras.layers.Conv2D(256, (3, 3), activation='relu', bias_initializer=point_one,
                                kernel_initializer=normalized_init(9)),

            keras.layers.Flatten(),

            tf.keras.layers.Dropout(rate=0.5),

            keras.layers.Dense(4096, activation='relu', bias_initializer=point_one,
                            # TODO why is smaller number better 216 vs 4096
                            kernel_initializer=normalized_init(216)),

            tf.keras.layers.Dropout(rate=0.5),
            keras.layers.Dense(4096, activation='relu', bias_initializer=point_one,
                            kernel_initializer=normalized_init(100)),

            keras.layers.Dense(Model.LEN_CLASS_NAMES, activation='softmax', bias_initializer='zeros',
                            kernel_initializer=normalized_init(100))  # 4096
        ])

        # learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=0.001, decay_steps=1200,
        #                                                                 end_learning_rate=0.0005*0.001, power=1)

        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),  # learning_rate=learning_rate_fn
                    loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)])

        return model

def main():
    train_sample, _ = Data.load()
    train_sample = train_sample.take(1)

    train_sample_gen = Preprocessing.create_generator(train_sample, for_training=True)
    model = Model.build()

    history = model.fit(x=train_sample_gen,
                    #steps_per_epoch = STEPS_PER_EPOCH,                    
                    #validation_data = test_ds, 
                    epochs=1)

if __name__ == '__main__':
    main()

# Here for testing (TODO Remove it)
a = main()
    