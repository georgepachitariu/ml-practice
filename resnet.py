import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, GlobalAveragePooling2D, Flatten, \
    Dropout, Dense, ReLU, Lambda
import tensorflow_addons as tfa
import pickle, urllib
from datetime import datetime
import os
import sys
import shutil



# TODO Adapt preprocessing
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
        image = tf.image.resize(image, size=tf.constant((256, 256)))
        return image, label

    @staticmethod
    # TODO TEMPORARY
    def _resize_testing(image: tf.Tensor, label: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        image = tf.image.resize(image, size=tf.constant((224, 224)))
        return image, label

        
    @staticmethod
    def _augment(image: tf.Tensor, label: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = tf.image.random_crop(image, size = (224, 224, 3))        
        image = tf.image.random_flip_left_right(image)

        image = tf.clip_by_value(image, -0.5, 0.5)

        return image, label

    # TODO
    # [PAPER] Our implementation for ImageNet follows the practice in [21, 41]. The image is resized with its shorter side ran-
    # domly sampled in [256, 480] for scale augmentation [41]. A 224×224 crop is randomly sampled from an image or its
    # horizontal flip, with the per-pixel mean subtracted [21]. The standard color augmentation in [21] is used.
    @staticmethod
    def create_generator(ds, for_training, batch_size = 128, buffer_size = 256):
        auto=tf.data.experimental.AUTOTUNE

        ds = ds.map(Preprocessing._split_dict, num_parallel_calls=auto)
        ds = ds.map(Preprocessing._normalize, num_parallel_calls=auto)
        if for_training:
            ds = ds.map(Preprocessing._resize, num_parallel_calls=auto)
        else:
            ds = ds.map(Preprocessing._resize_testing, num_parallel_calls=auto)
        
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
        #ds = ds.prefetch(buffer_size=1) # using "auto" ends up with OOM during validation step 

        return ds


# https://www.tensorflow.org/tutorials/customization/custom_layers
class ResnetIdentityBlock(tf.keras.Model):
  def __init__(self, input_filters, shortcuts_across_2_sizes=False, kernel_size=(3, 3)):
    super(ResnetIdentityBlock, self).__init__(name='')

    # [PAPER] (B) The projection shortcut in Eqn.(2) is used to match dimensions (done by 1×1 convolutions). 
    # For both options, when the shortcuts go across feature maps of two sizes, 
    # they are performed with a stride of 2.    
    if shortcuts_across_2_sizes:
        strides = 2
        self.shortcut_layer = Conv2D(input_filters*4, (1, 1), strides=strides) # size projection
    else:
        strides = 1
        self.shortcut_layer = Lambda(lambda x: x) # identity mapping

    self.conv2a = Conv2D(input_filters, (1, 1), strides=strides)
    self.bn2a = BatchNormalization()

    self.conv2b = Conv2D(input_filters, kernel_size, padding='same')
    self.bn2b = BatchNormalization()

    # the number of output filters is always the number of input filters * 4 
    self.conv2c = Conv2D(input_filters * 4, (1, 1))
    self.bn2c = BatchNormalization()


  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    x += self.shortcut_layer(input_tensor)
    return tf.nn.relu(x)

class Model:

    @staticmethod
    def build():
        # TODO [PAPER] We initialize the weights as in [13] and train all plain/residual nets from scratch.
        #point_zero_one = tf.compat.v1.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        
        # "L2 regularization is also called weight decay in the context of neural networks. 
        # Don't let the different name confuse you: weight decay is mathematically the exact same as L2 regularization."
        # https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
        # TODO
        #weight_decay = tf.keras.regularizers.l2(0)

        model = tf.keras.Sequential([
            # conv1
            # TODO Number of weights is ((11×11×3+1)×96) = 34944 where:
            #            11 * 11 = convolution filter size
            #                  3 = number of input layers 
            #                  1 = bias
            #                 96 = number of output layers
            Conv2D(64, (7, 7),  input_shape=(224, 224, 3), strides=2, 
                   # TODO are the next correct?
                   # bias_initializer=zero, 
                   # kernel_initializer=point_zero_one, kernel_regularizer=weight_decay
                   ),
            BatchNormalization(),
            ReLU(),

            # conv2_x
            # TODO Number of weights for 1 block is
            ResnetIdentityBlock(input_filters=64, shortcuts_across_2_sizes=True),
            ResnetIdentityBlock(input_filters=64),
            ResnetIdentityBlock(input_filters=64),

            # conv3_x
            ResnetIdentityBlock(input_filters=128, shortcuts_across_2_sizes=True),
            ResnetIdentityBlock(input_filters=128),
            ResnetIdentityBlock(input_filters=128),
            ResnetIdentityBlock(input_filters=128),

            # conv4_x
            ResnetIdentityBlock(input_filters=256, shortcuts_across_2_sizes=True),
            ResnetIdentityBlock(input_filters=256),
            ResnetIdentityBlock(input_filters=256),

            ResnetIdentityBlock(input_filters=256),
            ResnetIdentityBlock(input_filters=256),
            ResnetIdentityBlock(input_filters=256),

            # conv5_x
            ResnetIdentityBlock(input_filters=512, shortcuts_across_2_sizes=True),
            ResnetIdentityBlock(input_filters=512),
            ResnetIdentityBlock(input_filters=512),

            GlobalAveragePooling2D(),

            Flatten(),
            # 1000 categories
            Dense(1000, activation='softmax', 
                #bias_initializer=zero, kernel_initializer=point_zero_one
                ) 
        ])

        # Also check the ReduceLROnPlateau training callback lower
        model.compile(
                    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), 

                    # "categorical_crossentropy": uses a one-hot array to calculate the probability,
                    # "sparse_categorical_crossentropy": uses a category index
                    # source: https://stackoverflow.com/questions/58565394/what-is-the-difference-between-sparse-categorical-crossentropy-and-categorical-c
                    loss='sparse_categorical_crossentropy', 

                    metrics=['accuracy', 
                             tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
                             ])

        return model

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

class Resnet:
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

    # TODO is batch_size = 128 correct?
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

        # [PAPER]The learning rate starts from 0.1 and is divided by 10 when the error plateaus
        # TODO is min_lr=0.0001 necessary?
        # TODO [PAPER] and the models are trained for up to 60 × 10^4 iterations.\
        # TODO [PAPER] use a weight decay of 0.0001 and a momentum of 0.9.
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=0.0001)

        # Create a callback that saves the model's weights
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=Resnet._get_checkpoint_folder(version),
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
            path = Resnet._get_checkpoint_folder(version)
        self.model.load_weights(path)


if __name__ == '__main__':
    current_version = 'v0.1'
    network = Resnet()
    network.load_data(sample_fraction=0.5)
    network.create_generator(batch_size=32)
    network.build_model()
    
    print(network.model.summary())

    # The default behaviour should be to resume training for current version, 
    # so we don't make accidents by overwriting a trained model.
    #if len(sys.argv)==1 or sys.argv[1] is not 'start_new':
    #    network.load_model(current_version)

    network.train(dataset_iterations=20, version=current_version)

# TODO List
# [PAPER] In testing, for comparison studies we adopt the standard 10-crop testing [21]. For best results, we adopt the fully-
# convolutional form as in [41, 13], and average the scores at multiple scales (images are resized such that the shorter
# side is in {224, 256, 384, 480, 640}).


    
# Journal (Run log)

# Epoch 9/10
# 3750/3750 [==============================] - 841s 224ms/step - loss: 3.8735 - accuracy: 0.2156 - sparse_top_k_categorical_accuracy: 0.4435 - 
#                                                    val_loss: 4.3020 - val_accuracy: 0.1781 - val_sparse_top_k_categorical_accuracy: 0.3864 - lr: 0.0100



