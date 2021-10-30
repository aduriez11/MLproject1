from linear_regression import calculate_mse_e
import numpy as np

def compute_stoch_gradient(y, tx, w):
    """
    Compute a stochastic gradient
    :param y: ndarray: predicted values
    :param tx: ndarray: regressors
    :param w: ndarray: weights
    
    :return: tuple(ndarray,ndarray): stochastic gradient
    """
    e = y-tx@w
    stoc_grad = -1/len(y) * np.transpose(tx)@e
    return stoc_grad


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Split both labels and regressors to batches
    :param y: ndarray: predicted values
    :param tx: ndarray: regressors
    :param batch_size: int: size of the batch
    :param num_batches: int: number of batches
    :param shuffle: boolean: do permutations or not
    
    :return: tuple(ndarray,ndarray): batches of regressors and predicted values
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """
    Stochastic gradient descent algorithm
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


def StochasticGradientDescentSubmission(y, tXst,max_iters=100,gamma=0.001,batch_size=1):
    """
    Stochastic gradient descent for all the subsets
    :param y: list[ndarray]: list of predicted values for different subsets
    :param tXst: list[ndarray]: list of regressors for different subsets
    :param max_iter: int: maximum number of iterations during gradient descent
    :param gamma: float: step of gradient descent
    :param batch_size: int: size of batch   
    
    :return: tuple(list[ndarray],list[float]): list of best weights and losses (rmse) for each subset     
    """    
    # Initialization
    w_list = []
    loss_list = []
    
    # Start SGD for each subset
    for i in range(len(y)):
        w_initial = np.array([0 for i in range(tXst[i].shape[1])])
        sgd_losses, sgd_w = stochastic_gradient_descent(y[i], tXst[i], w_initial, batch_size, max_iters, gamma)
        w_list.append(sgd_w)
        loss_list.append(sgd_losses)
    return w_list, loss_list