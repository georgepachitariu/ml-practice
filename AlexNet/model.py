# TODO At test time, the network makes a prediction by extracting five 224 × 224 patches
# (the four corner patches and the center patch) as well as their horizontal reflections (hence ten patches in all),
# and averaging the predictions made by the network’s softmax layer on the ten patches

def normalized_init(number_inputs):
    return tf.compat.v1.keras.initializers.RandomNormal(mean=0.0, stddev=1 / number_inputs)

# TODO make it a class ?!

k_init = tf.compat.v1.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
point_one = tf.compat.v2.constant_initializer(value=0.01)

model = keras.Sequential([
    keras.layers.Conv2D(96, (11, 11), input_shape=(224, 224, 3), strides=4, activation='relu',
                        bias_initializer='zeros', kernel_initializer=normalized_init(121)),
    keras.layers.MaxPooling2D(pool_size=3, strides=2),

    keras.layers.Conv2D(256, (5, 5), activation='relu', bias_initializer=point_one,
                        kernel_initializer=normalized_init(25)),
    keras.layers.MaxPooling2D(pool_size=3, strides=2),

    keras.layers.Conv2D(384, (3, 3), activation='relu', bias_initializer='zeros',
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

    keras.layers.Dense(len(CLASS_NAMES), activation='softmax', bias_initializer='zeros',
                       kernel_initializer=normalized_init(100))  # 4096
])

# learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=0.001, decay_steps=1200,
#                                                                 end_learning_rate=0.0005*0.001, power=1)

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),  # learning_rate=learning_rate_fn
              loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)])