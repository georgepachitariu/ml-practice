import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from tensorflow import keras
import pickle, urllib
from datetime import datetime
import os
import shutil

class Data:
    @staticmethod
    def load() -> (tf.data.Dataset, tf.data.Dataset):
        train_ds, validation_ds = tfds.load(name="imagenet2012", split=['train', 'validation'],
                                            data_dir='/home/gpachitariu/SSD/data')
        
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

    @staticmethod
    def load_labelid_to_names():
        # Hacky way of getting the class names because I couldn't find them in the tensorflow dataset library.
        # More details here: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
        return pickle.load(urllib.request.urlopen('https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/'+
                                                  'd133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl') )


class Preprocessing:
    @staticmethod
    def _split_dict(t: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        return t['image'], t['label']

    # TODO is the type of the input tf.Tensor?
    @staticmethod
    def _normalize(image: tf.Tensor, label: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        # Change values range from [0, 255] to [-0.5, 0.5]
        image = (image / 255) - 0.5
        return image, label

    @staticmethod
    # this method is only used to reverse normalisation so we can display the images
    def denormalize(image: tf.Tensor) -> tf.Tensor:
        return (image + 0.5) * 255

    @staticmethod
    # TODO Resize: In the paper for images they kept the ratio, in mine the images were made square
    def _resize(image: tf.Tensor, label: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        image = tf.image.resize(image, size=tf.constant((224, 224)))
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
        image = tf.image.random_flip_left_right(image)

        image = tf.clip_by_value(image, -0.5, 0.5)

        return image, label

    @staticmethod
    def create_generator(ds, for_training, batch_size = 128, buffer_size = 256):
        auto=tf.data.experimental.AUTOTUNE

        ds = ds.map(Preprocessing._split_dict, num_parallel_calls=auto)
        ds = ds.map(Preprocessing._normalize, num_parallel_calls=auto)
        ds = ds.map(Preprocessing._resize, num_parallel_calls=auto)
        
        if for_training:
            ds = ds.map(Preprocessing._augment, num_parallel_calls=auto)
            ds = ds.repeat() # repeat forever
            ds = ds.shuffle(buffer_size=buffer_size)
        
        if batch_size > 1:
            ds = ds.batch(batch_size)

        # Prefetching overlaps the preprocessing and model execution of a training step. 
        # While the model is executing training step s, the input pipeline is reading the data for step s+1. 
        # Doing so reduces the step time to the maximum (as opposed to the sum) of the training and the time it takes to extract the data.
        # https://www.tensorflow.org/guide/data_performance
        ds = ds.prefetch(buffer_size=1) # using "auto" ends up with OOM during validation step 

        return ds

class Model:

    @staticmethod
    def build():

        # Following the paper: "We initialized the weights in each layer from a zero-mean Gaussian distribution with standard deviation 0.01."
        point_zero_one = tf.compat.v1.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)

        # "We initialized the neuron biases in the second, fourth, and fifth convolutional layers,
        # as well as in the fully-connected hidden layers, with the constant 1. This initialization accelerates
        # the early stages of learning by providing the ReLUs with positive inputs. We initialized the neuron
        # biases in the remaining layers with the constant 0."

        # I put this to 0.001 instead of 1, because with 1 it converges very slow in the beginning, (like the bias is so big that it "shadows" the true value)
        one = tf.compat.v2.constant_initializer(value=0.001) 
        zero = tf.compat.v2.constant_initializer(value=0)

        model = keras.Sequential([
            # 1st conv. layer
            # Number of weights is ((11×11×3+1)×96) = 34944 where:
            #            11 * 11 = convolution filter size
            #                  3 = number of input layers 
            #                  1 = bias
            #                 96 = number of output layers
            keras.layers.Conv2D(96, (11, 11),  input_shape=(224, 224, 3), strides=4, activation='relu', 
                                bias_initializer=zero, kernel_initializer=point_zero_one),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=3, strides=2),

            # 2nd conv. layer
            # Number of weights is ((5×5×96+1)×256) = 614656
            keras.layers.Conv2D(256, (5, 5), activation='relu', bias_initializer=one, kernel_initializer=point_zero_one),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=3, strides=2),

            # 3rd conv. layer
            keras.layers.Conv2D(384, (3, 3), activation='relu', bias_initializer=zero, kernel_initializer=point_zero_one),

            # 4th conv. layer
            keras.layers.Conv2D(384, (3, 3), activation='relu', bias_initializer=one, kernel_initializer=point_zero_one),

            # 5th conv. layer
            keras.layers.Conv2D(256, (3, 3), activation='relu', bias_initializer=one, kernel_initializer=point_zero_one),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=3, strides=2),
            
            keras.layers.Flatten(),
            keras.layers.Dropout(rate=0.5),

            keras.layers.Dense(4096, activation='relu', bias_initializer=one, kernel_initializer=point_zero_one), 
            keras.layers.Dropout(rate=0.5),

            keras.layers.Dense(4096, activation='relu', bias_initializer=one, kernel_initializer=point_zero_one),

            # 1000 categories
            keras.layers.Dense(1000, activation='softmax', bias_initializer=zero, kernel_initializer=point_zero_one) 
        ])


        # [PAPER] "We used an equal learning rate for all layers, which we adjusted manually throughout training."
        # Also check the ReduceLROnPlateau training callback lower
        model.compile(
                    # [PAPER] We trained our models using stochastic gradient descent with a batch size of 128 examples, momentum of 0.9, and weight decay of 0.0005.
                    # TODO this didn't work. In tensorboard I see that the kernel and bias distributions went to 0 after 5-10 epochs and the loss didn't decrease almost at all. 
                    # optimizer=tfa.optimizers.SGDW(weight_decay=0.0005, learning_rate = 0.01, momentum = 0.9),

                    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),  # learning_rate=learning_rate_fn

                    # "categorical_crossentropy": uses a one-hot array to calculate the probability,
                    # "sparse_categorical_crossentropy": uses a category index
                    # source: https://stackoverflow.com/questions/58565394/what-is-the-difference-between-sparse-categorical-crossentropy-and-categorical-c
                    loss='sparse_categorical_crossentropy', 

                    metrics=['accuracy', 
                             tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
                             ])

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

class Alexnet:
    

    def __init__(self):
        configure_gpu()

    def load_data(self, sample_fraction=1, only_one = False):
        # http://www.image-net.org/challenges/LSVRC/2012/
        # number_categories = 1000 
        # 1.2 million train images
        # 150 000 validation images
        total_train_data_size = 1.2 * 1000 * 1000 # The alternative of counting this would take ages: len(list(train_data))))
        total_validation_data_size = 150 * 1000

        print("Loading input dataset")
        train_data, validation_data = Data.load()

        self.train_data_size = int(sample_fraction * total_train_data_size)
        self.validation_data_size = int(sample_fraction * total_validation_data_size)

        if only_one: 
            # I use this in testing
            self.train_data_size = 1
            self.validation_data_size = 1

        print(f"A fraction of {sample_fraction} was selected from the total data")
        print(f"Number of examples in the Train dataset is {self.train_data_size} and in the Validation dataset is {self.validation_data_size}")    

        self.train_data = train_data.take(self.train_data_size)
        self.validation_data = validation_data.take(self.validation_data_size)

    def create_generator(self, batch_size = 128):
        print("Creating the generators")
        self.batch_size = batch_size
        self.train_augmented_gen = Preprocessing.create_generator(self.train_data, for_training=True, batch_size = self.batch_size)
        self.validation_gen = Preprocessing.create_generator(self.validation_data, for_training=False)
    
    def build_model(self):
        self.model = Model.build()

    @staticmethod
    def _get_checkpoint_folder(version) -> str:
        # Checkpoint files contain your model's weights
        # https://www.tensorflow.org/tutorials/keras/save_and_load        
        return 'trained_models/' + version + '/cp.ckpt'

    def train(self, dataset_iterations, version, logs='./logs'):

        # [PAPER] The heuristic which we followed was to divide the learning rate by 10 when the validation error
        # rate stopped improving with the current learning rate. The learning rate was initialized at 0.01 and
        # reduced three times prior to termination.
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001)

        # Create a callback that saves the model's weights
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=Alexnet._get_checkpoint_folder(version),
                                                 save_weights_only=True, verbose=1)

        # Create a callback that logs the progress so you can visualize it in Tensorboard
        log_dir = logs + '/fit/' + datetime.now().strftime('%Y%m%d-%H%M%S')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        print("Starting the training")
        self.history = self.model.fit( x=self.train_augmented_gen,
                            validation_data = self.validation_gen,
                            # An epoch is an iteration over the entire x and y data provided.
                            epochs = dataset_iterations,
                            # Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch
                            steps_per_epoch = self.train_data_size / self.batch_size,
                            callbacks=[reduce_lr, checkpoint_callback, tensorboard_callback]
                            )
    
    def predict(self, images):
        return self.model.predict(images)
    
    def load_model(self, version, path=None):
        self.build_model()
        if path is None:
            path = Alexnet._get_checkpoint_folder(version)
        self.model.load_weights(path)


if __name__ == '__main__':
    current_version = 'v1.2'
    network = Alexnet()
    network.load_data(sample_fraction=0.5)
    network.create_generator()
    network.build_model()

    # [PAPER] We trained the network for roughly 90 cycles through the
    # training set of 1.2 million images, which took five to six days on two NVIDIA GTX 580 3GB GPUs"
    network.train(dataset_iterations=45, version=current_version)

    
# Journal (Run log)
# New record
# Epoch 30/30:
# 938/937 [==============================] - 132s 141ms/step - loss: 2.6609 - accuracy: 0.3872 - sparse_top_k_categorical_accuracy: 0.6590 - 
# val_loss: 5.0430 - val_accuracy: 0.1736 - val_sparse_top_k_categorical_accuracy: 0.3643

# New record:
# sample_fraction=0.3 - Epoch 30/30
# 2812/2812 [============================>.] - ETA: 0s - loss: 2.8876 - accuracy: 0.3529 - sparse_top_k_categorical_accuracy: 0.6286

# [...]
