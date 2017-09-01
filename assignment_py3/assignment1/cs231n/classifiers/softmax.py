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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train):
      scores = X[i].dot(W)
      shift_scores = scores-max(scores) #考虑数值稳定性，将最大值平移到0，最终得出的结果仍旧一致
      loss_i = -shift_scores[y[i]] + np.log(sum(np.exp(shift_scores))) #计算损失函数
      loss += loss_i
      for j in range(num_classes):
          softmax_out = np.exp(shift_scores[j])/sum(np.exp(shift_scores)) #计算归一化概率值
          if j==y[i]:
              dW[:,j] += (-1+softmax_out)*X[i]
          else:
              dW[:,j] += softmax_out*X[i]

  loss = loss/num_train + 0.5*reg*np.sum(W*W)
  dW = dW/num_train+reg*W
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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  shift_scores = scores-np.max(scores,axis=1).reshape(-1,1) #max得到1*N，转为N*1
  softmax_out = np.exp(shift_scores)/np.sum(np.exp(shift_scores),axis=1).reshape(-1,1)  #计算各分类的概率,sum得到N*1

  loss = -np.sum(np.log(softmax_out[range(num_train),list(y)])) # 计算正确分类y[i]的损失值
  loss = loss/num_train + 0.5*reg*np.sum(W*W)

  dS = softmax_out.copy() 
  dS[range(num_train),list(y)] += -1
  dW = (X.T).dot(dS)
  dW = dW/num_train + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

