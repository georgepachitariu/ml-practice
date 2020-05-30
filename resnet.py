from data import Imagenet2012
import gpu
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from tensorflow.keras.layers import Conv2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, \
    Flatten, Dropout, Dense, ReLU, Lambda, Softmax
from tensorflow.dtypes import cast
from datetime import datetime
import os, sys, shutil


class Preprocessing:
    @staticmethod
    # this method is only used to reverse normalisation so we can display the images
    def denormalize(image: tf.Tensor) -> tf.Tensor:
        image = tf.clip_by_value(image, -0.5, 0.5)
        return (image + 0.5) * 255
        
    @staticmethod
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
    def _preprocess_both(t: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        image, label = t['image'], t['label']        
        image = (image / 255) - 0.5 # Normalize: change values range from [0, 255] to [-0.5, 0.5]
        return image, label

    @staticmethod
    # [PAPER] Our implementation for ImageNet follows the practice in [21, 41]. The image is resized with its shorter side 
    # randomly sampled in [256, 480] for scale augmentation [41]. A 224×224 crop is randomly sampled from an image or its horizontal flip, 
    # with the per-pixel mean subtracted [21]. The standard color augmentation in [21] is used.
    def _preprocess_train(image: tf.Tensor, label: tf.Tensor) -> (tf.Tensor, tf.Tensor):        
        image = Preprocessing.resize_same_ratio(image)  
        image = tf.image.random_flip_left_right(image)

        image = tf.image.random_crop(image, size = (224, 224, 3))        
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)        
        return image, label   

    @staticmethod
    def _preprocess_test(image: tf.Tensor, label: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        image = Preprocessing.resize_same_ratio(image, new_size=368)
        height = tf.shape(image)[0] 
        width = tf.shape(image)[1]

        s_images = []
        s_labels = []
        for im in [image, 
                   tf.image.flip_left_right(image)]:
            for start_height, start_width in [[0, 0],
                                            [0, width-224],
                                            [height-224, width-224],
                                            [height-224, 0],
                                            [height/2-224/2, width/2-224/2]
                                            ]:
                new_image_crop = tf.image.crop_to_bounding_box(image, offset_height=cast(start_height, tf.int32), 
                                                       offset_width=cast(start_width, tf.int32), 
                                                       target_height=224, target_width=224)
                s_images.append(new_image_crop)
                s_labels.append(label)

        return tf.stack(s_images), tf.stack(s_labels)
    
    @staticmethod
    def create_generator(ds, for_training, batch_size = 128, buffer_size = 256):
        auto=tf.data.experimental.AUTOTUNE
        
        if for_training:
            ds = ds.repeat()
            ds = ds.shuffle(buffer_size=buffer_size)

        ds = ds.map(Preprocessing._preprocess_both, num_parallel_calls=auto)

        if for_training: 
            ds = ds.map(Preprocessing._preprocess_train, num_parallel_calls=auto)    
            ds = ds.batch(batch_size)
        else:
            # all crops per image need to be in the same batch,
            # because later I do "Tensor segmentation" based on the labels in the current batch.
            # https://www.tensorflow.org/api_docs/python/tf/math#Segmentations
            ds = ds.map(Preprocessing._preprocess_test, num_parallel_calls=auto)
            ds = ds.unbatch() #  #size: ((10,height,width,channels), (10, 1), (10)); (stacks of 10 crops)
            # TODO I don't know why batch=300 works. It should be too big for memory.
            ds = ds.batch(300) # batch/stack them in multiples of 10 so that more of them fit the GPU
             

        # Prefetching overlaps the preprocessing and model execution of a training step. 
        # While the model is executing training step s, the input pipeline is reading the data for step s+1. 
        # Doing so reduces the step time to the maximum (as opposed to the sum) of the training and the time it takes to extract the data.
        # https://www.tensorflow.org/guide/data_performance
        #ds = ds.prefetch(buffer_size=1) # using "auto" ends up with OOM during validation step 
        return ds

    
    @staticmethod
    # this method is only used to reverse normalisation so we can display the images
    def denormalize(image: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        image = tf.clip_by_value(image, -0.5, 0.5)
        return (image + 0.5) * 255, label


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

class Resnet(tf.keras.Model):
    def __init__(self, version = 'v1.0'):
        super(Resnet, self).__init__(name='')
    
        self.version = version

        # TODO [PAPER] We initialize the weights as in [13] and train all plain/residual nets from scratch.
        #point_zero_one = tf.compat.v1.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        
        # "L2 regularization is also called weight decay in the context of neural networks. 
        # Don't let the different name confuse you: weight decay is mathematically the exact same as L2 regularization."
        # https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
        # TODO
        #weight_decay = tf.keras.regularizers.l2(0)

        # TODO bias_initializer, kernel_initializer, kernel_regularizer
        
        self.model = tf.keras.Sequential([
            # conv1
            # Number of weights is (7×7×3+1)×64 = 9472 where:
            #              7 * 7 = convolution filter size
            #                  3 = number of channels (input layers)
            #                  1 = bias
            #                 64 = number of output layers
            Conv2D(64, (7, 7),  strides=2, # input_shape=(224, 224, 3)
                   # TODO are the next correct?
                   # bias_initializer=zero, kernel_initializer=point_zero_one, kernel_regularizer=weight_decay
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

    def call(self, input_tensor, training=False):
        return self.model(input_tensor, training=training)

    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        images_crops, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        total_number_crops = tf.shape(images_crops)[0]
        distinct_images = total_number_crops / 10 # there are 10 crops per image
        segments = tf.repeat(tf.range(0, distinct_images, 1, dtype=tf.int32), repeats=10)

        y_pred_crops=self.model(images_crops, training=False)
        # I segment based on the label to get the mean score per 10 crops per image. Check the testing generator as well.
        y_pred = tf.math.segment_mean(y_pred_crops, segments)
        y = tf.math.segment_max(y, segments) # I segment y based on same rules to have same segmentation of y_pred and y

        # Updates stateful loss metrics.
        self.compiled_loss(
            y, y_pred, sample_weight, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def compile(self):
        super().compile(
            # Also check the ReduceLROnPlateau training callback lower
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), 
            # "categorical_crossentropy": uses a one-hot array to calculate the probability,
            # "sparse_categorical_crossentropy": uses a category index
            # source: https://stackoverflow.com/questions/58565394/what-is-the-difference-between-sparse-categorical-crossentropy-and-categorical-c
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy', 
                     tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)])

    @staticmethod
    def _get_checkpoint_folder(version) -> str:
        # Checkpoint files contain your model's weights
        # https://www.tensorflow.org/tutorials/keras/save_and_load        
        return 'trained_models/resnet' + version + '/cp.ckpt'

    def fit(self, x, validation_data, dataset_iterations, steps_per_epoch, logs1='./logs'):
        # [PAPER]The learning rate starts from 0.1 and is divided by 10 when the error plateaus
        # TODO [PAPER] and the models are trained for up to 60 × 10^4 iterations.\
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1)

        # Create a callback that saves the model's weights
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=Resnet._get_checkpoint_folder(self.version),
                                                 save_weights_only=True, verbose=1)

        # Create a callback that logs the progress so you can visualize it in Tensorboard
        log_dir = logs1 + '/fit/' + datetime.now().strftime('%Y%m%d-%H%M%S')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        print("Starting the training")
        return super().fit(x=x,
                              validation_data = validation_data,                            
                              epochs = dataset_iterations,
                              steps_per_epoch = steps_per_epoch,
                              callbacks=[reduce_lr, checkpoint_callback, tensorboard_callback]
                              )

    def load_weights(self, version, path=None):
        if path is None:
            path = Resnet._get_checkpoint_folder(version)
        self.model.load_weights(path)


def main():
    gpu.configure_gpu()

    train_data_size, validation_data_size, train_data, validation_data = Imagenet2012.load_data(sample_fraction=0.1, only_one=False)
    
    resnet = Resnet(version = 'v1.0')
    resnet.compile()

    batch_size=16
    print("Creating the generators")
    train_augmented_gen = Preprocessing.create_generator(train_data, for_training=True, batch_size = batch_size)
    validation_gen = Preprocessing.create_generator(validation_data, for_training=False)
    
    # The default behaviour should be to resume training for current version, 
    # so we don't make accidents by overwriting a trained model.
    #if len(sys.argv)==1 or sys.argv[1] is not 'start_new':
    #    network.load_model(current_version)
    history = resnet.fit(   x=train_augmented_gen,
                            validation_data=validation_gen,
                            # An epoch is an iteration over the entire x and y data provided.
                            dataset_iterations = 20,                   
                            # Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch
                            steps_per_epoch=train_data_size / batch_size )
    

if __name__ == '__main__':
    main()

# TODO [PAPER] use a weight decay of 0.0001
    
# Journal (Run log)

# Train time: 120000 / 16 = 7500 GPU steps
# Validation time: 15000 * 1 (10 crops) GPU steps 

# New record:
# sample_fraction=0.5: Epoch 15/20
# 18750/18750 [==] - 4127s 220ms/step - loss: 1.0002 - accuracy: 0.7436 - sparse_top_k_categorical_accuracy: 0.9163 - 
#                                   val_loss: 1.9326 - val_accuracy: 0.5734 - val_sparse_top_k_categorical_accuracy: 0.8046 - lr: 1.0000e-04

# Epoch 9/10
# 3750/3750 [==============================] - 841s 224ms/step - loss: 3.8735 - accuracy: 0.2156 - sparse_top_k_categorical_accuracy: 0.4435 - 
#                                                    val_loss: 4.3020 - val_accuracy: 0.1781 - val_sparse_top_k_categorical_accuracy: 0.3864 - lr: 0.0100

