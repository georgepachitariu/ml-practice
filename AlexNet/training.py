import tensorflow as tf


IMG_SIZE = 448  # TODO What should the Image size be?

# In the paper for images they kept the ratio, in mine the images were made square
def _resize(t: tf.Tensor) -> tf.Tensor:
    t['image'] = tf.image.resize(t['image'], (IMG_SIZE, IMG_SIZE))
    return t


# TODO It would be nice to have a peek at the first image
#    to see that the values are between -0.5 and 1
def _normalize(t: tf.Tensor) -> tf.Tensor:
    # the values are between 0 and 255. We make them between -0.5 and 0.5
    t['image'] = (t['image'] / 255) - 0.5
    return t


def _augment(t: tf.Tensor) -> tf.Tensor:
    # These 4 (rotation, brightness, contrast, flip)
    # are added by me as a helper to get to 70% accuracy, not part of the paper
    # TODO img = tf.keras.preprocessing.image.random_rotation(rg=45, fill_mode='constant', cval=0)
    t['image'] = tf.image.random_brightness(t['image'], max_delta=0.1)
    t['image'] = tf.image.random_contrast(t['image'], lower=0.9, upper=1.1)

    crop_size = (tf.random.uniform([1])[0] * 0.25 + 0.75) * IMG_SIZE
    # zoom in & out. max(zoom_out)=original size
    # t['image'] = tf.image.random_crop(t['image'], size = (crop_size, crop_size))
    t['image'] = tf.image.resize(t['image'], size=tf.constant((224, 224)))

    t['image'] = tf.image.random_flip_left_right(t['image'])
    return t


train_image_count = tf.data.experimental.cardinality(train_ds).numpy()
BATCH_SIZE = 128  # TODO is batch size correct?
STEPS_PER_EPOCH = np.ceil(train_image_count / BATCH_SIZE)


# TODO "prepare" is a very vague keyword. Find a more specific one
def prepare(ds, is_train, shuffle_buffer_size=1000,
            auto=tf.data.experimental.AUTOTUNE):
    ds = ds.map(resize, num_parallel_calls=auto)
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    if is_train:
        ds = ds.repeat()  # Repeat forever
        ds = ds.map(augment, num_parallel_calls=auto)
    # TODO should normalize be after augment?
    # Because normalize is in testing as well, and we don't have augment there.
    ds = ds.map(normalize, num_parallel_calls=auto)

    ds = ds.batch(BATCH_SIZE)

    # dataset fetches batches in the background while the model is training.
    ds = ds.prefetch(buffer_size=auto)

    return ds