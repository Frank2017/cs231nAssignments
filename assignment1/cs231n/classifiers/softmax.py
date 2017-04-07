import numpy as np
from random import shuffle

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
  num_train, dim = X.shape
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
      scores = X[i].dot(W)  #(1,C)
      #   logC = -np.max(scores)
      #   scores = scores + logC
      scores_exp = np.exp(scores)
      loss += -1.0 * scores[y[i]] + np.log(np.sum(scores_exp))
      # calculate dW
      for j in xrange(num_classes):
          if j == y[i]:
              dW[:,y[i]] += -1.0 * X[i].T + (1 / np.sum(scores_exp)) * scores_exp[y[i]] * X[i].T
          else:
              dW[:,j] += (1 / np.sum(scores_exp)) * scores_exp[j] * X[i].T
              pass
      pass
  pass
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

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
  num_train, dim = X.shape
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)  #(N,C)
  scores_y = scores[np.arange(num_train), y]
  scores_exp = np.exp(scores)
  loss += (np.sum(-1.0 * scores_y) + np.sum(np.log(np.sum(np.exp(scores), axis=1))))
  loss /= num_train
  loss += np.sum(W * W)
  # calculate dW
  # dlogdW = 1 / (np.sum(scores_exp, axis=1).reshape(num_train, 1))  #(N,1)
  # dlogdW_NbyC = dlogdW.dot(np.ones(num_classes).reshape(1, num_classes))  #(N,C)
  # dLdW = dlogdW_NbyC * scores_exp  #(N,C)
  # dLdW[np.arange(num_train), y] += -1  #(N,C)
  # dW = X.T.dot(dLdW)
  # dW /= num_train
  # dW += reg * W

  P = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)
  p = np.zeros_like(P)
  p[np.arange(num_train), y] = 1
  dW = (1.0/num_train) * X.T.dot((P-p))
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
