from cross_validation import *
from polynomial_regression import build_poly,calculate_mse
import numpy as np

def ridge_regression(y, tx, lambda_):
    """
    Build ridge regression with least squares method
    :param y: ndarray: predicted values
    :param tx: ndarray: regressors
    :param lambda_: float: penalizing coefficient 
    
    :return: ndarray: weights of the model
    """
    aI = 2 * len(y) * lambda_ * np.identity(tx.shape[1])
    A = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(A,b)
    return w

def cross_validation_ridge(y, x, k_indices, k, lambda_, degree):
    """
    Cross-validate ridge regression
    :param y: ndarray: predicted values
    :param x: ndarray: regressors
    :param k_indices: ndarray: list of indexes for the batch
    :param k: int: number of the batch
    :param lambda_: float: penalizing coefficient
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

    w = ridge_regression(y_train, x_trainpoly, lambda_)
    
    loss_tr = np.sqrt(2*calculate_mse(y_train, x_trainpoly, w))
    loss_te= np.sqrt(2*calculate_mse(y_test, x_testpoly, w))
  
    return loss_tr, loss_te


def RidgeRegressionSubmission(y,tXst,num_degrees=5,k_fold=4,seed=1,lambdas=np.logspace(-2,1,8)):
    """
    Ridge polynomial regression for all the subsets with cross_validation
    :param y: list[ndarray]: list of predicted values for different subsets
    :param tXst: list[ndarray]: list of regressors for different subsets
    :param num_degrees: int: number of degrees from 0 to num_degree for cross-validation
    :param k_fold: int: number of folds in cross-validation
    :param seed: int: seed for random number generator
    :param lambdas: ndarray: list of different lambdas for cross-validation
    
    :return: tuple(list[int],list[float]): list of best degrees and losses (rmse) for each subset     
    """
    # define lists to store the loss of training data and test data
    matrix_te0=np.zeros(shape=(num_degrees,len(lambdas)))
    matrix_tr0=np.zeros(shape=(num_degrees,len(lambdas)))
    matrix_te1=np.zeros(shape=(num_degrees,len(lambdas)))
    matrix_tr1=np.zeros(shape=(num_degrees,len(lambdas)))
    matrix_te2=np.zeros(shape=(num_degrees,len(lambdas)))
    matrix_tr2=np.zeros(shape=(num_degrees,len(lambdas)))
    matrix_te3=np.zeros(shape=(num_degrees,len(lambdas)))
    matrix_tr3=np.zeros(shape=(num_degrees,len(lambdas)))
    #loop through degrees
    for degree in np.arange(num_degrees):
        #loop through different lambdas
        for ind_lmbd,lambda_ in enumerate(lambdas):
            for i in range(0,len(y)):
                k_indices,indices = build_k_indices(y[i], k_fold, seed)
                loss_tr=0
                loss_te=0
                for k in range(k_fold):
                    l_tr, l_te = cross_validation_ridge(y[i], tXst[i], k_indices, k, lambda_, degree)
                    loss_tr+=l_tr
                    loss_te+=l_te
                if i==0:
                    matrix_te0[degree][ind_lmbd]=loss_te/k_fold
                    matrix_tr0[degree][ind_lmbd]=loss_tr/k_fold
                elif i==1:
                    matrix_te1[degree][ind_lmbd]=loss_te/k_fold
                    matrix_tr1[degree][ind_lmbd]=loss_tr/k_fold
                elif i==2:
                    matrix_te2[degree][ind_lmbd]=loss_te/k_fold
                    matrix_tr2[degree][ind_lmbd]=loss_tr/k_fold
                elif i==3:
                    matrix_te3[degree][ind_lmbd]=loss_te/k_fold
                    matrix_tr3[degree][ind_lmbd]=loss_tr/k_fold  
    res_degrees=[]
    res_lambdas=[]
    res_losses=[]
    
    #get the best degree lambda couple  
    result = np.where(matrix_te0 == np.amin(matrix_te0))
    listOfCoordinates = list(zip(result[0], result[1]))
    best_degree = listOfCoordinates[0][0]
    best_lambda = listOfCoordinates[0][1]
    print('Best parameters for 0 subset are:',best_degree,lambdas[best_lambda],' with error: ',matrix_te0[best_degree][best_lambda])
    res_degrees.append(best_degree)
    res_lambdas.append(best_lambda)
    res_losses.append(matrix_te0[best_degree][best_lambda])
    
    result = np.where(matrix_te1 == np.amin(matrix_te1))
    listOfCoordinates = list(zip(result[0], result[1]))
    best_degree = listOfCoordinates[0][0]
    best_lambda = listOfCoordinates[0][1]
    print('Best parameters for 1 subset are:',best_degree,lambdas[best_lambda],' with error: ',matrix_te1[best_degree][best_lambda])
    res_degrees.append(best_degree)
    res_lambdas.append(best_lambda)
    res_losses.append(matrix_te1[best_degree][best_lambda])
    
    result = np.where(matrix_te2 == np.amin(matrix_te2))
    listOfCoordinates = list(zip(result[0], result[1]))
    best_degree = listOfCoordinates[0][0]
    best_lambda = listOfCoordinates[0][1]
    print('Best parameters for 2 subset are:',best_degree,lambdas[best_lambda],' with error: ',matrix_te2[best_degree][best_lambda])
    res_degrees.append(best_degree)
    res_lambdas.append(best_lambda)
    res_losses.append(matrix_te2[best_degree][best_lambda])
    
    result = np.where(matrix_te3 == np.amin(matrix_te3))
    listOfCoordinates = list(zip(result[0], result[1]))
    best_degree = listOfCoordinates[0][0]
    best_lambda = listOfCoordinates[0][1]
    print('Best parameters for 3 subset are:',best_degree,lambdas[best_lambda],' with error: ',matrix_te3[best_degree][best_lambda])
    res_degrees.append(best_degree)
    res_lambdas.append(best_lambda)
    res_losses.append(matrix_te3[best_degree][best_lambda])
    
    return res_degrees,res_lambdas,res_losses