# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
from past.builtins import xrange
from .neural_net import TwoLayerNet


class TwoLayerNetImproved(TwoLayerNet):
    """
    Two-layer NN with leakyReLU activation function
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        super().__init__(input_size, hidden_size, output_size, std)
        # Xavier initializer for ReLU
        # self.params['W1'] = np.random.randn(input_size, hidden_size) / np.sqrt(input_size / 2)
        # self.params['W2'] = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size / 2)


    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # Perform the forward pass, computing the class scores for the input.       #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        layer1_mul = X.dot(W1)
        layer1_bias = layer1_mul + b1
        layer1_activation = np.maximum(layer1_bias, 0.01 * layer1_bias)
        layer2_mul = layer1_activation.dot(W2)
        layer2_bias = layer2_mul + b2
        scores = layer2_bias
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # Finish the forward pass, and compute the loss. This should include        #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        log_c_coeff = np.max(scores, axis=1)
        corrected_logits = scores - np.reshape(log_c_coeff, (-1, 1))
        logits_exp = np.exp(corrected_logits)
        sum_logits_exp = np.sum(logits_exp, axis=1)
        log_sum_logits_exp = np.log(sum_logits_exp)
        inverted_true_logits = -1 * corrected_logits[np.arange(N), y]
        current_loss = inverted_true_logits + log_sum_logits_exp
        sum_loss = np.sum(current_loss)
        loss = sum_loss / N
        loss += reg * np.sum(np.square(W1))
        loss += reg * np.sum(np.square(W2))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # Compute the backward pass, computing the derivatives of the weights       #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        d_corrected_logits = np.zeros_like(corrected_logits)
        d_inverted_true_logits = np.ones_like(inverted_true_logits)
        d_log_sum_logits_exp = np.ones_like(log_sum_logits_exp)
        d_corrected_logits[np.arange(N), y] += -1 * d_inverted_true_logits
        d_sum_logits_exp = d_log_sum_logits_exp * (1 / sum_logits_exp)
        d_logits_exp = np.zeros_like(corrected_logits)  # distribute gradient
        d_logits_exp[:, ::1] = np.reshape(d_sum_logits_exp, (-1, 1))
        d_corrected_logits += d_logits_exp * logits_exp
        d_logits = d_corrected_logits
        d_logits[np.arange(N), np.argmax(scores, axis=1)] -= np.sum(d_logits, axis=1)

        grads["W2"] = layer1_activation.T.dot(d_logits)
        grads["b2"] = np.sum(d_logits, axis=0)
        d_layer1_activation = d_logits.dot(W2.T) * (
            (layer1_activation > 0).astype(np.float64) + (layer1_activation < 0).astype(np.float64) * 0.01)
        grads["W1"] = X.T.dot(d_layer1_activation)
        grads["b1"] = np.sum(d_layer1_activation, axis=0)

        grads["W2"] /= N
        grads["W2"] += 2 * reg * W2
        grads["b2"] /= N
        grads["W1"] /= N
        grads["W1"] += 2 * reg * W1
        grads["b1"] /= N

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads