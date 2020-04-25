import tensorflow_datasets as tfds

def load() -> (tf.data.Dataset, tf.data.Dataset):
    train_ds, validation_ds = tfds.load(name="imagenet2012", split=['train', 'validation'],
                                        data_dir='/home/gpachitariu/HDD/data')
    return train_ds, validation_ds
