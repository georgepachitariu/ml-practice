import tensorflow as tf
import tensorflow_datasets as tfds

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
