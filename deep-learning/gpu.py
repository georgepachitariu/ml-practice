import tensorflow as tf

def configure_gpu():
    # from https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:                
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)