import unittest
import numpy as np
import supportvectormachine_gpu as svm_gpu
import supportvectormachine as svm


class SupportVectorMachineTests(unittest.TestCase):

    def test_regression_gpugradient_numpygradient(self):

        input = np.array([[1]])
        weights = np.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9]])
        y=np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
        hinge_loss_term=np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        gradient1 = svm_gpu.SupportVectorMachineGPU.compute_gradient_analitically(input, weights, y,
                                                                      hinge_loss_term, regularization_value=0)

        gradient2 = svm.SupportVectorMachine.compute_gradient_analitically(input, weights, y,
                                                                      hinge_loss_term, regularization_value=0)

        assert np.all(gradient1 == gradient2)
