import numpy as np

class Tools:
    # TODO test it
    @staticmethod
    def add_bias(x_train):
        (m, n) = x_train.shape
        tmp = np.ones((m, n + 1), dtype=np.float16)
        tmp[:, 1:] = x_train
        return tmp