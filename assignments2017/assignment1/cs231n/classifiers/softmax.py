import numpy as np


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    # num_features = X.shape[1]
    num_classes = W.shape[1]
    for i in range(num_train):
        current_logits = X[i].dot(W)
        # loss += -np.log(np.exp(current_logits[y[i]]) / np.sum(np.exp(current_logits)))
        ###
        log_c_coeff = np.max(current_logits)
        corrected_logits = current_logits - log_c_coeff  # correction to avoid large numbers
        logits_exp = np.exp(corrected_logits)
        sum_logits_exp = np.sum(logits_exp)
        log_sum_logits_exp = np.log(sum_logits_exp)
        inverted_true_logit = -1 * corrected_logits[y[i]]
        current_loss = inverted_true_logit + log_sum_logits_exp
        ###
        loss += current_loss

        ###
        d_corrected_logits = np.zeros_like(current_logits)
        d_inverted_true_logit = 1.0
        d_log_sum_logits_exp = 1.0
        d_corrected_logits[y[i]] += -1 * d_inverted_true_logit
        d_sum_logits_exp = d_log_sum_logits_exp * (1 / sum_logits_exp)
        d_logits_exp = np.full_like(corrected_logits, d_sum_logits_exp)  # distribute gradient
        d_corrected_logits += d_logits_exp * logits_exp

        d_current_logits = d_corrected_logits
        d_current_logits[np.argmax(current_logits)] -= np.sum(d_current_logits)
        current_dW = np.zeros_like(dW)
        # for j in range(num_features):
        #     for k in range(num_classes):
        #         current_dW[j, k] = d_current_logits[k] * X[i, j]
        for k in range(num_classes):
            current_dW[:, k] = d_current_logits[k] * X[i]
        ###
        dW += current_dW

    dW /= num_train
    loss /= num_train

    # regularization
    loss += reg * np.sum(np.square(W))
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]
    logits = X.dot(W)
    log_c_coeff = np.max(logits, axis=1)
    corrected_logits = logits - np.reshape(log_c_coeff, (-1, 1))
    logits_exp = np.exp(corrected_logits)
    sum_logits_exp = np.sum(logits_exp, axis=1)
    log_sum_logits_exp = np.log(sum_logits_exp)
    inverted_true_logits = -1 * corrected_logits[np.arange(num_train), y]
    current_loss = inverted_true_logits + log_sum_logits_exp
    sum_loss = np.sum(current_loss)
    loss = sum_loss / num_train

    d_corrected_logits = np.zeros_like(corrected_logits)
    d_inverted_true_logits = np.ones_like(inverted_true_logits)
    d_log_sum_logits_exp = np.ones_like(log_sum_logits_exp)
    d_corrected_logits[np.arange(num_train), y] += -1 * d_inverted_true_logits
    d_sum_logits_exp = d_log_sum_logits_exp * (1 / sum_logits_exp)
    d_logits_exp = np.zeros_like(corrected_logits)  # distribute gradient
    d_logits_exp[:, ::1] = np.reshape(d_sum_logits_exp, (-1, 1))
    d_corrected_logits += d_logits_exp * logits_exp

    d_logits = d_corrected_logits
    d_logits[np.arange(num_train), np.argmax(logits, axis=1)] -= np.sum(d_logits, axis=1)

    dW = X.T.dot(d_logits)
    dW /= num_train
    # regularization
    loss += reg * np.sum(np.square(W))
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
