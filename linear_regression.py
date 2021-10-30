import numpy as np

def calculate_mse_e(e):
    """
    Calculate the mse for vector e
    :param e: float: error value
    
    :return: float: mean squared error
    """
    return 1/2*np.mean(e**2)

def least_squares(y, tx):
    """
    Calculate the least squares solution
    :param y: ndarray: predicted values
    :param tx: ndarray: regressors
    
    :return: tuple(ndarray,float):weights and loss as mse value
    """
    A = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)
    e = y - tx.dot(w)
    return w, calculate_mse_e(e)

def LinearRegressionSubmission(y, tXst):
    """
    Build linear regression for all the subsets
    :param y: list[ndarray]: list of predicted values for different subsets
    :param tXst: list[ndarray]: list of regressors for different subsets
    
    :return: tuple(list[ndarray],list[float]): list of weights and losses (mse) for each subset
    """
    w_list = []
    rmse_list = []
    for i in range(len(y)):
        w, mse = least_squares(y[i], tXst[i])
        w_list.append(w)
        rmse_list.append(np.sqrt(2*mse))
    return w_list, rmse_list
