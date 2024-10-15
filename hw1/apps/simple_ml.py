"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl



def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(image_filename, "rb") as img_file:
        magic_num, img_num, row, col = struct.unpack(">4i", img_file.read(16))
        assert(magic_num == 2051)
        tot_pixels = row * col
        X = np.vstack([np.array(struct.unpack(f"{tot_pixels}B", img_file.read(tot_pixels)), dtype=np.float32) for _ in range(img_num)])
        X -= np.min(X)
        X /= np.max(X)

    with gzip.open(label_filename, "rb") as label_file:
        magic_num, label_num = struct.unpack(">2i", label_file.read(8))
        assert(magic_num == 2049)
        y = np.array(struct.unpack(f"{label_num}B", label_file.read()), dtype=np.uint8)

    return X, y
    ### END YOUR CODE


def softmax_loss(Z, y_one_hot):
    batch_size = Z.shape[0]
    lhs = ndl.log(ndl.exp(Z).sum(axes=(1, )))
    rhs = (Z * y_one_hot).sum(axes=(1, ))
    loss = (lhs - rhs).sum()
    return loss / batch_size


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    batch_cnt = (X.shape[0] + batch - 1) // batch
    num_classes = W2.shape[1]
    one_hot_y = np.eye(num_classes)[y]
    for batch_idx in range(batch_cnt):
        start_idx = batch_idx * batch
        end_idx = min(X.shape[0], (batch_idx+1)*batch)
        X_batch = X[start_idx:end_idx, :]
        y_batch = one_hot_y[start_idx:end_idx]
        X_tensor = ndl.Tensor(X_batch)
        y_tensor = ndl.Tensor(y_batch) 
        first_logits = X_tensor @ W1 # type: ndl.Tensor
        first_output = ndl.relu(first_logits) # type: ndl.Tensor
        second_logits = first_output @ W2 # type: ndl.Tensor
        loss_err = softmax_loss(second_logits, y_tensor) # type: ndl.Tensor
        loss_err.backward()
        
        new_W1 = ndl.Tensor(W1.numpy() - lr * W1.grad.numpy())
        new_W2 = ndl.Tensor(W2.numpy() - lr * W2.grad.numpy())
        W1, W2 = new_W1, new_W2

    return W1, W2


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
