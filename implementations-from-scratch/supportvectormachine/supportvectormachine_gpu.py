import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time

class SupportVectorMachineGPU:

    # TODO Test it
    @staticmethod
    def compute_gradient_analitically(input, weights, y, hinge_loss_term, regularization_value):
        start = time.time()

        # The row is the input entry for which the loss was computed for;
        # The column is the label weight vector, for which the loss was computed for;
        # The loss is a scalar;
        assert np.ndim(hinge_loss_term) == 2

        # As you see in this method, if hinge_loss_term < 0, the gradient is not affected anymore by the
        # result of the "hinge_loss_term". "It doesn'T try to strecth/separate the 2 class labels"
        hinge_loss_term[hinge_loss_term < 0] = 0
        # If hinge_loss_term[i,j] > 0, it means that for row i and label j, the label is incorrect and
        # it had a higher predicted value than the correct label. These are the "support vectors".
        # Support vectors are a few input entries that are used to train the model (separate the classes).
        hinge_loss_term[hinge_loss_term > 0] = 1

        input = input.astype(np.float32)
        y = y.astype(np.int32)
        gradient = np.zeros_like(weights, dtype=np.float32)
        hinge_loss_term = hinge_loss_term.astype(np.int32)

        # TODO In the C code it is possible to access elements with 2 dimensions (input[i][j]) but I couldn't make it work
        mod = SourceModule("""
          __global__ void compute_gradient( float *input_batch, int *y_batch, int *hinge_loss_term_batch, 
          float* gradient, int number_entries, int number_features, int number_labels)
          {
            int weight_index_k = blockIdx.x * blockDim.x + threadIdx.x; 
            int label_col_j = blockIdx.y * blockDim.y + threadIdx.y;
            
            if(weight_index_k >= number_features)
            {
                return; // index out of bounds
            }
            
            for(int entry_row_i=0; entry_row_i < number_entries; entry_row_i++)
            {
                // if the label is not the correct one:
                if (y_batch[entry_row_i * number_labels + label_col_j] != 1)  // prediction for an incorrect label
                {
                    // ib: for the incorrect label, we add single losses multiplied by the input entry:
                    gradient[label_col_j * number_features + weight_index_k] +=
                        hinge_loss_term_batch[entry_row_i * number_labels + label_col_j] * 
                            input_batch[entry_row_i * number_features + weight_index_k]; 
                }
                else  // prediction for the correct label
                {
                    int count = 0;
                    for (int label_col_j_2 = 0; label_col_j_2 < number_labels; label_col_j_2++)
                    {
                        if (label_col_j_2 != label_col_j && 
                            hinge_loss_term_batch[entry_row_i * number_labels + label_col_j_2] == 1)
                        {
                            count += 1;
                        }
                    }
                    // 1a: for the correct label, we make the sum of all the incorrect label an multiply if with the input row
                    gradient[label_col_j * number_features + weight_index_k] -= count * input_batch[entry_row_i * number_features + weight_index_k];
                }
            }      
          }
        """)

        number_features = np.int32(input.shape[1])
        number_labels = np.int32(weights.shape[0])

        func = mod.get_function("compute_gradient")
        batch_size = 256
        features_per_block = min(100, input.shape[1])

        for batch_i in range(0, input.shape[0], batch_size):
            real_batch_size = batch_size if batch_i + batch_size < input.shape[0] \
                                    else input.shape[0] - batch_i

            func(cuda.In(np.copy(input[batch_i: batch_i + real_batch_size])),
                 cuda.In(y[batch_i: batch_i + real_batch_size]),
                 cuda.In(hinge_loss_term[batch_i: batch_i + real_batch_size]),
                 cuda.InOut(gradient),
                 np.int32(real_batch_size), number_features, number_labels,
                 block=(features_per_block, 10, 1),
                 grid=(int(np.ceil(number_features / features_per_block)),
                       1, 1)
                 )

        gradient /= np.count_nonzero(np.nansum(hinge_loss_term, axis=1) != 0)  # normalization

        # TODO test with weights squared
        gradient += regularization_value * weights

        return gradient



