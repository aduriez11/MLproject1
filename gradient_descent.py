from linear_regression import calculate_mse_e
import numpy as np

def compute_gradient(y, tx, w):
    """
    Compute the gradient
    :param y: ndarray: predicted values
    :param tx: ndarray: regressors
    :param w: ndarray:weights
    
    :return: tuple(ndarray,ndarray): gradient and error
    """
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """
    Gradient descent algorithm
    :param y: ndarray: predicted values
    :param tx: ndarray: regressors
    :param initial_w: ndarray: initial weights
    :param max_iters: int: maximum number of iterations
    :param gamma: float: step of gradient descent
    
    :return: tuple(float,ndarray): last loss and weights
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        #loss = np.sqrt(2*calculate_mse(err))
        loss = np.sqrt(2*calculate_mse_e(err))
        # update w by gradient descent
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)

    return losses[-1], ws[-1]


def GradientDescentSubmission(y, tXst, max_iters=1000, gamma=0.5):
    """
    Gradient descent for all the subsets
    :param y: list[ndarray]: list of predicted values for different subsets
    :param tXst: list[ndarray]: list of regressors for different subsets
    :param max_iters: int: maximum number of iterations
    :param gamma: float: step of gradient descent
    
    :return: tuple(list[ndarray],list[float]): list of best weights and losses (mse) for each subset     
    """
    # Initialization
    w_list = []
    loss_list = []
    
    # Start GD for each subset
    for i in range(len(y)):
        w_initial = np.array([np.random.rand() for i in range(tXst[i].shape[1])])
        gradient_loss, gradient_w = gradient_descent(y[i], tXst[i], w_initial, max_iters, gamma)
        w_list.append(gradient_w)
        loss_list.append(gradient_loss)
    return w_list, loss_list