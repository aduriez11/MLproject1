from logistic_regression import sigmoid,calculate_loss_logistic,calculate_gradient_logistic
from cross_validation import build_k_indices
import numpy as np

def penalized_logistic_regression(y, tx, w, lambda_):
    """
    Calculate penalized logistic loss and gradient
    :param y: ndarray: predicted values
    :param tx: ndarray: regressors
    :param w: ndarray: weights
    :param lambda_: float: penalizing coefficient
    
    :return: tuple(float,ndarray): loss and gradient vector for penalized logistic regression
    """
    loss = calculate_loss_logistic(y, tx, w) + lambda_ * (w.T.dot(w))
    gradient = calculate_gradient_logistic(y, tx, w) + 2 * lambda_ * w
    return loss, gradient

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    One step of penalized logistic gradient descent
    :param y: ndarray: predicted values
    :param tx: ndarray: regressors
    :param w: ndarray: weights
    :param gamma: float: step of gradient descent
    :param lambda_: float: penalizing coefficient
    
    :return: tuple(float,ndarray): loss for the step and updated weight vector
    """  
    loss,gradient = penalized_logistic_regression(y,tx,w,lambda_)
    w = w-gradient*gamma
    return loss, w

def cross_validation_logistic_regression(y, x, k_indices, k, gamma,lambda_):
    """
    Cross-validate penalized logistic regression
    :param y: ndarray: predicted values
    :param x: ndarray: regressors
    :param k_indices: ndarray: list of indexes for the batch
    :param k: int: number of the batch
    :param gamma: float: step of gradient descent
    :param lambda_: float: penalizing coefficient
    
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

    w = np.random.rand(x.shape[1], 1)
    _,w = learning_by_penalized_gradient(y_train, x_train, w, gamma, lambda_)

    loss_tr = calculate_loss_logistic(y_train, x_train, w)
    loss_te= calculate_loss_logistic(y_test, x_test, w)

    return loss_tr, loss_te

def PenalizedLogisticRegressionSubmission(y, tXst, max_iter=10000,threshold=1e-8,k_fold = 1,seed=1,gammas = np.arange(0.000001),lambdas =   np.arange(0.01)):
    """
    Penalized logistic regression for all the subsets with cross_validation
    :param y: list[ndarray]: list of predicted values for different subsets
    :param tXst: list[ndarray]: list of regressors for different subsets
    :param max_iters: int: maximum number of iterations
    :param threshold: float: bound to stop gradient descent
    :param k_fold: int: number of folds in cross-validation
    :param seed: int: seed for random number generator
    :param gammas: ndarray: list of gammas (step of gradient descent) for cross-validation
    :param lambdas: ndarray: list of lambdas (penalized coefficient) fot cross-validation 
    
    :return: tuple(list[float],list[float],list[float]): best gammas, lambdas and losses for each subset
    """
    all_losses=[]
    weights=[]

    matrix_te0=np.zeros(shape=(len(gammas),len(lambdas)))
    matrix_te1=np.zeros(shape=(len(gammas),len(lambdas)))
    matrix_te1=np.zeros(shape=(len(gammas),len(lambdas)))
    matrix_te1=np.zeros(shape=(len(gammas),len(lambdas)))

    for i in range(0,len(y)):
        k_indices,indices = build_k_indices(y[i], k_fold, seed)
        losses = []
        loss_tr = []
        loss_te = []
        # build tx and ty
        tx = np.c_[np.ones((y[i].shape[0], 1)), tXst[i]]
        ty=y[i].reshape(y[i].shape[0],1)
        
        print('Training for subset {0}\n'.format(i))
        # start the logistic regression
        for ind_gm,gamma in enumerate(gammas):
            for ind_lmbd,lambda_ in enumerate(lambdas):
                for k in range(k_fold):
                    loss_tr=0
                    loss_te=0
                # get loss and update w.
                    for iter in range(max_iter):
                        l_tr,l_te = cross_validation_logistic_regression(ty, tx, k_indices, k, gamma,lambda_)
                        loss_tr+=l_tr
                        loss_te+=l_te
                    # log info
                    if iter % 100 == 0:
                        print("Current iteration={i}, loss={l}".format(i=iter, l=np.squeeze(loss)))
                      # converge criterion
                    if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                        break
                    if i==0:
                        matrix_te0[ind_gm][ind_lmbd]=loss_te/k_fold
                    elif i==1:
                        matrix_te1[ind_gm][ind_lmbd]=loss_te/k_fold
                    elif i==2:
                        matrix_te2[ind_gm][ind_lmbd]=loss_te/k_fold
                    elif i==3:
                        matrix_te3[ind_gm][ind_lmbd]=loss_te/k_fold
    res_gammas=[]
    res_lambdas=[]
    res_losses=[]
    
    result = np.where(matrix_te0 == np.amin(matrix_te0))
    listOfCoordinates = list(zip(result[0], result[1]))
    best_gamma = listOfCoordinates[0][0]
    best_lambda = listOfCoordinates[0][1]
    print('Best parameters for 0 subset are:',gammas[best_gamma],lambdas[best_lambda],' with error: ',matrix_te0[best_gamma][best_lambda])
    res_gammas.append(best_gamma)
    res_lambdas.append(best_lambda)
    res_losses.append(matrix_te0[best_gamma][best_lambda])
    
    result = np.where(matrix_te1 == np.amin(matrix_te1))
    listOfCoordinates = list(zip(result[0], result[1]))
    best_gamma = listOfCoordinates[0][0]
    best_lambda = listOfCoordinates[0][1]
    print('Best parameters for 1 subset are:',gammas[best_gamma],lambdas[best_lambda],' with error: ',matrix_te1[best_gamma][best_lambda])
    res_gammas.append(best_gamma)
    res_lambdas.append(best_lambda)
    res_losses.append(matrix_te0[best_gamma][best_lambda])
    
    result = np.where(matrix_te2 == np.amin(matrix_te2))
    listOfCoordinates = list(zip(result[0], result[1]))
    best_gamma = listOfCoordinates[0][0]
    best_lambda = listOfCoordinates[0][1]
    print('Best parameters for 2 subset are:',gammas[best_gamma],lambdas[best_lambda],' with error: ',matrix_te2[best_gamma][best_lambda])
    res_gammas.append(best_gamma)
    res_lambdas.append(best_lambda)
    res_losses.append(matrix_te0[best_gamma][best_lambda])
    
    result = np.where(matrix_te3 == np.amin(matrix_te3))
    listOfCoordinates = list(zip(result[0], result[1]))
    best_gamma = listOfCoordinates[0][0]
    best_lambda = listOfCoordinates[0][1]
    print('Best parameters for 3 subset are:',gammas[best_gamma],lambdas[best_lambda],' with error: ',matrix_te3[best_gamma][best_lambda])
    res_gammas.append(best_gamma)
    res_lambdas.append(best_lambda)
    res_losses.append(matrix_te0[best_gamma][best_lambda])
    
    return res_gammas,res_lambdas,res_losses