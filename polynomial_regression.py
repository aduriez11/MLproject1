from linear_regression import least_squares
from cross_validation import *
import numpy as np

def calculate_mse(y, tx, w):
    """
    Calculate the mse when given predicted values, regressor and weights
    :param y: ndarray: predicted values
    :param tx: ndarray: regressors
    :param w: ndarray:weights
    
    :return: float: mean squared error
    """
    e = y - tx.dot(w)
    return 1/2*np.mean(e**2)

def build_poly(x, degree):
    """
    Polynomial basis function for input data x, for j=0 up to j=degree
    :param x: ndarray: regressors
    :param degree: int: degree of polynomial
    
    :return: ndarray: polynomial regressors
    """
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def cross_validation_leastsquares(y, x, k_indices, k, degree):
    """
    Cross-validate polynomial regression
    :param y: ndarray: predicted values
    :param x: ndarray: regressors
    :param k_indices: ndarray: list of indexes for the batch
    :param k: int: number of the batch
    :param degree: int: degree of polynomial regression
    
    :return: tuple(list[float],list[float]): list of losses (rmse) for train and test    
    """
    x_test = x[k_indices[k]]
    y_test = y[k_indices[k]]
    x_train = []
    y_train = []

    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    x_train = x[tr_indice]
    y_train = y[tr_indice]

    x_testpoly = build_poly(x_test,degree)
    x_trainpoly = build_poly(x_train,degree)
  
    w,mse = least_squares(y_train, x_trainpoly)
    loss_tr = np.sqrt(2*calculate_mse(y_train, x_trainpoly, w))
    loss_te= np.sqrt(2*calculate_mse(y_test, x_testpoly, w))
  
    return loss_tr, loss_te

def PolynomialRegressionSubmission(y,tXst,degrees=[i for i in range(12)],k_fold=4,seed=1):
    """
    Polynomial regression based on least squares for all the subsets with cross_validation
    :param y: list[ndarray]: list of predicted values for different subsets
    :param tXst: list[ndarray]: list of regressors for different subsets
    :param degrees: list[int]: list of possible degrees for cross-validation in polynomial regression
    :param k_fold: int: number of folds in cross-validation
    :param seed: int: seed for random number generator
    
    :return: tuple(list[int],list[float]): list of best degrees and losses (rmse) for each subset     
    """
    mean_rmse = np.zeros((4,len(degrees)))
    rmse_tr = []
    rmse_te = []
    test = []
    deg = []
    minimum = []
    
    #loop through degrees
    for ind,degree in enumerate(degrees):
        #loop through subsets
        for i in range(0,len(y)):
            #split data into several batches for cross-validaton
            k_indices,indices = build_k_indices(y[i], k_fold,seed)
            
            loss_te=0
            rmse_te_tmp = []
            #loop through batches
            for k in range(k_fold):
                l_tr, l_te = cross_validation_leastsquares(y[i], tXst[i], k_indices, k,degree)
                loss_te+=l_te
            #storing mean loss across all batches
            mean_rmse[i][ind]=loss_te/k_fold
    #printing best parameters
    for i in range(0,len(y)):
        test.append(mean_rmse[i].tolist())
        deg.append(degrees[np.argmin(mean_rmse[i])])
        minimum.append(min(test[i]))
        print("Best degree for subset {i}: {degree}, test error: {err}".format(i = i,degree=degrees[np.argmin(mean_rmse[i])],err=min(test[i])))
    return deg , minimum