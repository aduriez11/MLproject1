import numpy as np
from gradient_descent import compute_gradient

def least_squares_GD(y, tx, initial w, max_iters, gamma):
    """
    Compute linear regression using gradient descent
    :param y: ndarray: predicted values
    :param tx: ndarray: regressors
    :param initial_w: ndarray: initial weights
    :param max_iters: int: maximum number of iterations
    :param gamma: float: step of gradient descent
    
    :return: tuple(float,ndarray): loss and weights
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)

        loss = np.sqrt(2*calculate_mse_e(err))
        # update w by gradient descent
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)

    return losses[-1], ws[-1]

from stochastic_gradient_descent import compute_stoch_gradient,batch_iter

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """
    Compute linear regression using stochastic gradient descent
    :param y: ndarray: predicted values
    :param tx: ndarray: regressors
    :param initial_w: ndarray: initial weights
    :param batch_size: int: size of batch    
    :param max_iters: int: maximum number of iterations
    :param gamma: float: step of gradient descent
    
    :return: tuple(float,ndarray): last loss and weights
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for batcy,batcx in batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
            stoc_grad = compute_stoch_gradient(batcy,batcx,w)
            error=y-tx.dot(w)
            loss = np.sqrt(2*calculate_mse_e(error))
            w = w-gamma*stoc_grad
            ws.append(w)
            losses.append(loss)
    return losses[-1], ws[-1]

from linear_regression import calculate_mse_e

def least_squares(y, tx):
    """
    Calculate the least squares solution
    :param y: ndarray: predicted values
    :param tx: ndarray: regressors
    
    :return: tuple(float,ndarray): loss and weights
    """
    A = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)
    e = y - tx.dot(w)
    return calculate_mse_e(e),w

from polynomial_regression import calculate_mse

def ridge_regression(y, tx, lambda_):
    """
    Build ridge regression with least squares method
    :param y: ndarray: predicted values
    :param tx: ndarray: regressors
    :param lambda_: float: penalizing coefficient 
    
    :return: tuple(float,ndarray): loss and weights
    """
    aI = 2 * len(y) * lambda_ * np.identity(tx.shape[1])
    A = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(A,b)
    loss=np.sqrt(2*calculate_mse(y, tx, w))
    
    return loss, w

from logistic_regression import learning_by_gradient_descent

def logistic_regression (y, tXst, initial_w,max_iters, gamma):
    """
    Logistic regression
    :param y: ndarray: predicted values
    :param tXst: ndarray: regressors
    :param initial_w: ndarray: initial vector of weights
    :param max_iters: int: maximum number of iterations
    :param gamma: float: step of gradient descent
    
    :return: tuple(float,ndarray): loss and weights   
    """
    threshold=1e-6
    
    # build tx and ty
    tx = np.c_[np.ones((y.shape[0], 1)), tXst]
    w = initial_w
    ty=y.reshape(y.shape[0],1)

    # start the logistic regression
    for iter in range(max_iters):    
        # get loss and update w
        loss, w = learning_by_gradient_descent(ty, tx, w, gamma)
            
        #store loss
        losses.append(loss)
            
        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return loss,w

from penalized_logistic_regression import learning_by_penalized_gradient

def reg_logistic_regression (y, tXst,lambda_,initial_w,max_iters, gamma):
    """
    Penalized logistic regression
    :param y: ndarray: predicted values
    :param tXst: ndarray: regressors
    :param lambda_: float: penalized coefficient
    :param initial_w: ndarray: initial vector of weights
    :param max_iters: int: maximum number of iterations
    :param gamma: float: step of gradient descent
    
    :return: tuple(float,ndarray): loss and weights   
    """
    threshold = 1e-6

    # build tx and ty
    tx = np.c_[np.ones((y.shape[0], 1)), tXst]
    ty=y.reshape(y.shape[0],1)
    w=initial_w

    for iter in range(max_iters):
    
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(ty, tx, w, gamma, lambda_)
        
        #store loss
        losses.append(loss)
        
        #converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return loss,w