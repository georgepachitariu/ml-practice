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
    # Resize: In the paper for images they kept the ratio, in mine the images were made square

    IMG_SIZE = 224
    BUFFER_SIZE = 1000

    @staticmethod
    def _split_dict(t: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        return t['image'], t['label']

    # TODO is the type of the input tf.Tensor?
    @staticmethod
    def _normalize(image: tf.Tensor, label: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        # Change values range from [0, 255] to [-0.5, 0.5]
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
        # crop_size = (tf.random.uniform([1])[0] * 0.25 + 0.75) * Preprocessing.IMG_SIZE
        # zoom in & out. max(zoom_out)=original size
        # image = tf.image.random_crop(image, size = (crop_size, crop_size))
        image = tf.image.resize(image, size=tf.constant((224, 224)))
        image = tf.image.random_flip_left_right(image)

        return image, label

    @staticmethod
    def create_generator(ds, for_training, batch_size = 128, dataset_repeat = 5):
        auto=tf.data.experimental.AUTOTUNE

        ds = ds.map(Preprocessing._split_dict, num_parallel_calls=auto)
        ds = ds.map(Preprocessing._normalize, num_parallel_calls=auto)
        
        ds = ds.shuffle(buffer_size=Preprocessing.BUFFER_SIZE)

        if for_training:
            ds = ds.repeat(count=dataset_repeat)
            ds = ds.map(Preprocessing._augment, num_parallel_calls=auto)
        ds = ds.batch(batch_size)

        # dataset fetches batches in the background while the model is training.
        ds = ds.prefetch(buffer_size=auto)

        return ds

class Model:

    @staticmethod
    def build(number_categories):

        # Following the paper: "We initialized the weights in each layer from a zero-mean Gaussian distribution with standard deviation 0.01."
        point_zero_one = tf.compat.v1.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)

        # "We initialized the neuron biases in the second, fourth, and fifth convolutional layers,
        # as well as in the fully-connected hidden layers, with the constant 1. This initialization accelerates
        # the early stages of learning by providing the ReLUs with positive inputs. We initialized the neuron
        # biases in the remaining layers with the constant 0."
        one = tf.compat.v2.constant_initializer(value=1)
        zero = tf.compat.v2.constant_initializer(value=0)


        model = keras.Sequential([

            # 1st conv. layer
            # Number of weights is ((11×11×3+1)×96) = 34944 where:
            #            11 * 11 = convolution filter size
            #                  3 = number of input layers 
            #                  1 = bias
            #                 96 = number of output layers
            keras.layers.Conv2D(96, (11, 11),  input_shape=(224, 224, 3), strides=4, activation='relu', 
                                bias_initializer=zero,
                                kernel_initializer=point_zero_one),
            keras.layers.MaxPooling2D(pool_size=3, strides=2),

            # 2nd conv. layer
            # Number of weights is ((5×5×96+1)×256) = 614656
            keras.layers.Conv2D(256, (5, 5), activation='relu', bias_initializer=one, kernel_initializer=point_zero_one),
            keras.layers.MaxPooling2D(pool_size=3, strides=2),

            # 3rd conv. layer
            keras.layers.Conv2D(384, (3, 3), activation='relu', bias_initializer=zero, kernel_initializer=point_zero_one),

            # 4th conv. layer
            keras.layers.Conv2D(384, (3, 3), activation='relu', bias_initializer=one, kernel_initializer=point_zero_one),

            # 5th conv. layer
            keras.layers.Conv2D(256, (3, 3), activation='relu', bias_initializer=one, kernel_initializer=point_zero_one),
            keras.layers.Flatten(),
            tf.keras.layers.Dropout(rate=0.5),

            keras.layers.Dense(4096, activation='relu', bias_initializer=zero, kernel_initializer=point_zero_one), 
            tf.keras.layers.Dropout(rate=0.5),

            keras.layers.Dense(4096, activation='relu', bias_initializer=zero, kernel_initializer=point_zero_one),

            keras.layers.Dense(number_categories, activation='softmax', bias_initializer='zeros',
                            kernel_initializer=point_zero_one) 
        ])

        
        # TODO 2. From paper: "We used an equal learning rate for all layers, which we adjusted manually throughout training.
        # The heuristic which we followed was to divide the learning rate by 10 when the validation error
        # rate stopped improving with the current learning rate. The learning rate was initialized at 0.01 and
        # reduced three times prior to termination. We trained the network for roughly 90 cycles through the
        # training set of 1.2 million images, which took five to six days on two NVIDIA GTX 580 3GB GPUs"

        # TODO: What is weight decay? Add weight decay.
        # learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=0.001, decay_steps=1200,
        #                                                                 end_learning_rate=0.0005*0.001, power=1)
        

        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),  # learning_rate=learning_rate_fn
                    loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)])

        return model

# TODO From paper: "At test time, the network makes a prediction by extracting five 224 × 224 patches
# (the four corner patches and the center patch) as well as their horizontal reflections (hence ten patches in all),
# and averaging the predictions made by the network’s softmax layer on the ten patches

def configure_gpu():
    # from https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def run(number_categories=1000, dataset_repeat=5,  sample_fraction=1):
    configure_gpu()

    # http://www.image-net.org/challenges/LSVRC/2012/
    # number_categories = 1000 
    # 1.2 million train images
    # 150 000 validation images
    len_train_data = 1.2 * 1000 * 1000 # The alternative of counting this would take ages: len(list(train_data))))
    len_validation_data = 150 * 1000

    print("Loading input dataset")
    train_data, validation_data = Data.load()

    train_sample_size = int(sample_fraction * len_train_data)
    validation_sample_size = int(sample_fraction * len_validation_data)
    print(f"Number of examples in the Train sample is {train_sample_size} and in the Validation sample is {validation_sample_size}")    
    sample_train_data = train_data.take(train_sample_size)
    sample_validation_data = validation_data.take(validation_sample_size)
    
    print("Creating the generators")
    train_sample_gen = Preprocessing.create_generator(sample_train_data, for_training=True, 
                                                      batch_size = 32,
                                                      dataset_repeat=dataset_repeat)
    validation_sample_gen = Preprocessing.create_generator(sample_validation_data, for_training=False)

    model = Model.build(number_categories)
    print("Starting the training")
    history = model.fit( x=train_sample_gen,
                         validation_data = validation_sample_gen,
                         # An epoch is an iteration over the entire x and y data provided.
                         epochs = 10,
                         # Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch
                         steps_per_epoch = 5
                       )
    return history

if __name__ == '__main__':
    run(dataset_repeat=5, sample_fraction=0.1)
