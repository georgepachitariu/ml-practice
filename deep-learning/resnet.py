from data import Imagenet2012
import gpu
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, GlobalAveragePooling2D, \
    Flatten, Dropout, Dense, ReLU, Lambda, Softmax
from tensorflow.dtypes import cast
from datetime import datetime
import os, sys, shutil


class Preprocessing:        
    @staticmethod
    # this resizes the images so that the smallest side is either new_size or a random value between 256 and 480
    def resize_same_ratio(image: tf.Tensor, new_size=None, random_size=True) -> tf.Tensor:
        height_old = cast(tf.shape(image)[0], tf.float64)  
        width_old = cast(tf.shape(image)[1], tf.float64)
        shorter_side = tf.minimum(width_old, height_old)

        if random_size:
            new_size = tf.random.uniform(shape=[], minval=256, maxval=480, dtype=tf.float64)
        else:
            new_size = new_size
        scaling_factor = shorter_side / new_size
        height_new = tf.cast(height_old / scaling_factor, tf.int32)
        width_new = tf.cast(width_old / scaling_factor, tf.int32)
        
        return tf.image.resize(image, size=tf.stack([height_new, width_new]))
    
    @staticmethod
    # this method is only used to reverse normalisation so we can display the images
    def denormalize(image: tf.Tensor) -> tf.Tensor:
        image = tf.clip_by_value(image, -0.5, 0.5)
        return (image + 0.5) * 255

    @staticmethod
    # [PAPER] Our implementation for ImageNet follows the practice in [21, 41]. The image is resized with its shorter side 
    # randomly sampled in [256, 480] for scale augmentation [41]. A 224×224 crop is randomly sampled from an image or its horizontal flip, 
    # with the per-pixel mean subtracted [21]. The standard color augmentation in [21] is used.
    def _augment_train(image: tf.Tensor, label: tf.Tensor) -> (tf.Tensor, tf.Tensor):        
        image = Preprocessing.resize_same_ratio(image)  
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_crop(image, size = (224, 224, 3))        

        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)        
        #image = tf.image.random_saturation(image, lower=0.9, upper=1.1)

        return image, label   

    @staticmethod
    def _prepare_testing_10crops(image: tf.Tensor, label: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        image = Preprocessing.resize_same_ratio(image, new_size=368) 
        height = tf.shape(image)[0] 
        width = tf.shape(image)[1]

        s_images = []
        s_labels = []
        for im in [image, tf.image.flip_left_right(image)]:
            # The ten crops are the left, upper, right, bottom & center (and their mirrors)
            for offset_height, offset_width in [[height/2-112, 0], # left
                                            [0, width/2-112], # upper
                                            [height/2-112, width-224], # right
                                            [height-224, width/2-112], # bottom
                                            [height/2-112, width/2-112] # center
                                            ]:
                new_image_crop = tf.image.crop_to_bounding_box(im, offset_height=cast(offset_height, tf.int32), 
                                                       offset_width=cast(offset_width, tf.int32), 
                                                       target_height=224, target_width=224)
                s_images.append(new_image_crop)
                s_labels.append(label)

        return tf.stack(s_images), tf.stack(s_labels)
    
    @staticmethod
    def create_generator(ds, for_training, batch_size, buffer_size = 10000):
        auto=tf.data.experimental.AUTOTUNE
        
        if for_training:
            ds = ds.shuffle(buffer_size=buffer_size)
            ds = ds.repeat()

        # Normalize & change the data type: change values range from [0, 255] to [-0.5, 0.5]
        ds = ds.map(lambda r: (tf.dtypes.cast(r['image']/255-0.5, dtype=tf.float32), 
                               r['label']), 
                    num_parallel_calls=auto)

        if for_training: 
            ds = ds.map(Preprocessing._augment_train, num_parallel_calls=auto)    
            ds = ds.batch(batch_size)
        else:
            # All 10 crops per image need to be in the same batch,
            # because later I do Tensor segmentation based on the labels in the current batch.
            # https://www.tensorflow.org/api_docs/python/tf/math#Segmentations
            ds = ds.map(Preprocessing._prepare_testing_10crops, num_parallel_calls=auto)
            ds = ds.unbatch().batch(60) # Batch them in a multiple of 10 so that more of them fit the GPU

        # Prefetching overlaps the preprocessing and model execution of a training step. 
        # While the model is executing training step s, the input pipeline is reading the data for step s+1. 
        # Doing so reduces the step time to the maximum (as opposed to the sum) of the training and the time it takes to extract the data.
        # https://www.tensorflow.org/guide/data_performance
        ds = ds.prefetch(buffer_size=1) # using "auto" ends up with OOM during validation step 
        return ds

class Weights:
    @staticmethod
    def init():
        # Following [13] I initialized the weights zero-mean Gaussian distribution whith standard deviation (std) = sqrt(2/n) 
        #   and bias=0, where n=number of input units in the weight tensor.
        # "A proper initialization method should avoid reducing or magnifying the magnitudes of input signals exponentially."
        return tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')

def weight_decay():
    # "L2 regularization is also called weight decay in the context of neural networks. 
    # Don't let the different name confuse you: weight decay is mathematically the exact same as L2 regularization."
    # https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
    # [PAPER] use a weight decay of 0.0001
    return tf.keras.regularizers.l2(0.0001)

class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, input_filters, name, first_conv_stride=1, shortcuts_across_2_sizes=False):
        super(ResnetIdentityBlock, self).__init__(name=name)

        # [PAPER] The projection shortcut in Eqn.(2) is used to match dimensions (done by 1×1 convolutions). 
        # For both options, when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2.    
        if shortcuts_across_2_sizes:
            self.shortcut_layer = tf.keras.Sequential([
                Conv2D(input_filters*4, (1, 1), strides=first_conv_stride, 
                    kernel_initializer=Weights.init(), kernel_regularizer=weight_decay(),
                    use_bias=False, name='res'+name+'_branch1'), # size projection

                BatchNormalization(name='bn'+name+'_branch1')
            ])

        else:
            self.shortcut_layer = Lambda(lambda x: x) # identity mapping

        self.conv2a = Conv2D(input_filters, (1, 1), strides=first_conv_stride, 
                             kernel_initializer=Weights.init(), kernel_regularizer=weight_decay(),
                             use_bias=False, 
                             name='res'+name+'_branch2a')
        self.bn2a = BatchNormalization(name='bn'+name+'_branch2a')
        self.relu2a = ReLU(name='res'+name+'_branch2a_relu')

        self.conv2b = Conv2D(input_filters, (3, 3), strides=1, padding='same', 
                             kernel_initializer=Weights.init(), kernel_regularizer=weight_decay(),
                             use_bias=False)
        self.bn2b = BatchNormalization(name='bn'+name+'_branch2b')
        self.relu2b = ReLU(name='res'+name+'_branch2b_relu')

        # number of output filters = number of input filters * 4 
        self.conv2c = Conv2D(input_filters * 4, (1, 1), strides=1,
                             kernel_initializer=Weights.init(), kernel_regularizer=weight_decay(),
                             use_bias=False)
        self.bn2c = BatchNormalization(name='bn'+name+'_branch2c')

        self.relu = ReLU(name='res'+name+'_relu')


    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = self.relu2a(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = self.relu2b(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += self.shortcut_layer(input_tensor)
        return self.relu(x)

class Resnet(tf.keras.Model):
    def __init__(self, version):
        super(Resnet, self).__init__(name='')
    
        self.version = version
        
        # The naming of layers is consistent with the naming from here:
        # http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006
        
        self.model = tf.keras.Sequential([
            # conv1
            # Number of weights is (7×7×3+1)×64 = 9472 where:
            #              7 * 7 = convolution filter size
            #                  3 = number of channels (input layers)
            #                  1 = bias
            #                 64 = number of output layers
            Conv2D(64, (7, 7),  strides=2, input_shape=(224, 224, 3), padding='same', 
                   kernel_initializer=Weights.init(), kernel_regularizer=weight_decay(),
                   bias_initializer='zeros', name='conv1'),
            BatchNormalization(name='bn_conv1'),
            ReLU(name='conv1_relu'),
            MaxPool2D(pool_size=(3, 3), strides=2, padding='same', name='pool1'),

            ResnetIdentityBlock(input_filters=64, name='2a', first_conv_stride=1, shortcuts_across_2_sizes=True),
            ResnetIdentityBlock(input_filters=64, name='2b'),
            ResnetIdentityBlock(input_filters=64, name='2c'),

            ResnetIdentityBlock(input_filters=128, name='3a', first_conv_stride=2, shortcuts_across_2_sizes=True),
            ResnetIdentityBlock(input_filters=128, name='3b'),
            ResnetIdentityBlock(input_filters=128, name='3c'),
            ResnetIdentityBlock(input_filters=128, name='3d'),

            ResnetIdentityBlock(input_filters=256, name='4a', first_conv_stride=2, shortcuts_across_2_sizes=True),
            ResnetIdentityBlock(input_filters=256, name='4b'),
            ResnetIdentityBlock(input_filters=256, name='4c'),
            ResnetIdentityBlock(input_filters=256, name='4d'),
            ResnetIdentityBlock(input_filters=256, name='4e'),
            ResnetIdentityBlock(input_filters=256, name='4f'),

            ResnetIdentityBlock(input_filters=512, name='5a', first_conv_stride=2, shortcuts_across_2_sizes=True),
            ResnetIdentityBlock(input_filters=512, name='5b'),
            ResnetIdentityBlock(input_filters=512, name='5c'),

            GlobalAveragePooling2D(name='pool5'),

            Flatten(),            
            Dense(1000, activation='softmax', 
                  kernel_initializer=Weights.init(), kernel_regularizer=weight_decay(),
                  bias_initializer='zeros', name='fc1000') # 1000 categories
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

    def get_checkpoint_folder(self) -> str:
        return 'trained_models/resnet_' + self.version

    def get_checkpoint_file(self) -> str:
        return self.get_checkpoint_folder() + "/{epoch:02d}"

    def fit(self, x, validation_data, dataset_iterations, initial_epoch, steps_per_epoch, lr_fn, logs1='./logs'):
        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_fn)

        # Callback that saves the model's weights: https://www.tensorflow.org/tutorials/keras/save_and_load
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.get_checkpoint_file(),                                                  
                                                 save_weights_only=True, verbose=1)  #save_best_only=False
        
        # Callback that logs the progress so you can visualize it in Tensorboard.
        log_dir = logs1 + '/fit/' + self.version
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=2)

        print("Starting the training")
        return super().fit(x=x,
                              validation_data = validation_data,                            
                              initial_epoch = initial_epoch,
                              epochs = dataset_iterations,
                              steps_per_epoch = steps_per_epoch,
                              callbacks=[learning_rate_scheduler, checkpoint_callback, tensorboard_callback]
                              )

    def get_latest_checkpoint(self):
        f = self.get_checkpoint_folder()
        weights_file = tf.train.latest_checkpoint(f)
        epoch = int(weights_file.split('/')[-1])
        return weights_file, epoch

        

def main():
    gpu.configure_gpu()
    
    version = 'v2.4-2020-July-26'
    initial_epoch = 7 # initial_epoch will be 1 more than this
    resume_training = False
        
    def lr_fn(epoch):
        # This is manually tuned. I let it run more to see where the training error plateaus, 
        # and then came back to pick the right epoch to switch to a smaller leaning rate.
        if epoch < 6: return 0.1
        elif epoch < 50: return 0.01
        else: return 0.001

    r = Resnet(version=version)
    r.compile()
    r.build(input_shape=(None, 224, 224, 3))
    
    # By default it will resume training by loading the weights from the previous completed epoch (checkpoint).
    if resume_training:
        weights_file, initial_epoch = r.get_latest_checkpoint()
        r.load_weights(weights_file)
    elif initial_epoch == 0:
        pass # this is the start of training so there are no files.
    else:
        weights_file = r.get_checkpoint_file().format(epoch=initial_epoch)
        r.load_weights(weights_file)
     
    print(r.model.summary())

    batch_size=64
    print("Creating the generators")
    train_data_size, validation_data_size, train_data, validation_data = Imagenet2012.load_data(sample_fraction=1, only_one=False)
    train_augmented_gen = Preprocessing.create_generator(train_data, for_training=True, batch_size=batch_size)
    validation_gen = Preprocessing.create_generator(validation_data, for_training=False, 
                    batch_size=None # batch_size is treated differently during validations
                    )
    
    history = r.fit(x=train_augmented_gen,
                         validation_data=validation_gen,
                         initial_epoch=initial_epoch, 
                         dataset_iterations=80,                   
                         # steps_per_epoch = total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch
                         steps_per_epoch=train_data_size/batch_size,
                         lr_fn=lr_fn)    
    
if __name__ == '__main__':
    main()

# Journal (Run log)
# Better training:
# Epoch 73/80
#       loss: 0.7877 - accuracy: 0.8024 - sparse_top_k_categorical_accuracy: 0.9327 - 
#       val_loss: 1.2672 - val_accuracy: 0.7195 - val_sparse_top_k_categorical_accuracy: 0.8930

# New record
# Epoch 75/80
#       loss: 0.9106 - accuracy: 0.7748 - sparse_top_k_categorical_accuracy: 0.9192 - 
#       val_loss: 1.2568 - val_accuracy: 0.7201 - val_sparse_top_k_categorical_accuracy: 0.8947

# Pretrained:
#       loss: 0.9174 - accuracy: 0.7747 - sparse_top_k_categorical_accuracy: 0.9223 - 
#       val_loss: 1.0786 - val_accuracy: 0.7738 - val_sparse_top_k_categorical_accuracy: 0.9329

# New record (full dataset):
# Epoch 37/50
# 8775s loss: 1.3503 - accuracy: 0.6819 - sparse_top_k_categorical_accuracy: 0.8658 - 
#       val_loss: 1.4499 - val_accuracy: 0.7074 - val_sparse_top_k_categorical_accuracy: 0.8866 - lr: 1.0000e-07
