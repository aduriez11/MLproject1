import numpy as np

def sigmoid(x):
    """
    Calculate stable sigmoid function
    :param x: ndarray: input parameter for sigmoid function
    
    :return: ndarray: vector after implementation sigmoid function
    """
    t=np.copy(x)
    t[t>0]=1/(1+np.exp(-t[t>0]))
    t[t<0]=np.exp(t[t<0])/(1+np.exp(t[t<0]))
    return t

def calculate_loss_logistic(y, tx, w):
    """ 
    Calculate loss for logistic regression
    :param y: ndarray: predicted values
    :param tx: ndarray: regressors
    :param w: ndarray: weights
    
    :return: float: logistic regression loss
    """
    pred = sigmoid(tx.dot(w))
    loss=((y+1)/2.0).T.dot(np.log(pred+1e-20)) + (1 - ((y+1)/2.0)).T.dot(np.log(1 - pred+1e-20))
    return np.squeeze(loss)

def calculate_gradient_logistic(y, tx, w):
    """
    Calculate gradient for logistic regression
    :param y: ndarray: predicted values
    :param tx: ndarray: regressors
    :param w: ndarray: weights
    
    :return: ndarray: gradiet vector for logistic regression
    """
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred-(y+1)/2.0)
    return grad

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    One step of logistic gradient descent
    :param y: ndarray: predicted values
    :param tx: ndarray: regressors
    :param w: ndarray: weights
    :param gamma: float: step of gradient descent
    
    :return: tuple(float,ndarray): loss for the step and updated weight vector
    """
    loss=calculate_loss_logistic(y,tx,w)
    grad = calculate_gradient_logistic(y,tx,w)
    w = w-gamma*grad
    return loss, w

def LogisticRegressionSubmission(y, tXst, max_iter=10000,threshold=1e-6,gamma=0.00001):
    """
    Logistic regression for all the subsets
    :param y: list[ndarray]: list of predicted values for different subsets
    :param tXst: list[ndarray]: list of regressors for different subsets
    :param max_iter: int: maximum number of iterations
    :param threshold: float: bound to stop gradient descent
    :param gamma: float: step of gradient descent
    
    :return: tuple(list[int],list[float]): list of best degrees and losses (rmse) for each subset     
    """
    all_losses=[]
    weights=[]
    
    for i in range(0,len(y)):
        losses = []
        
        # build tx and ty
        tx = np.c_[np.ones((y[i].shape[0], 1)), tXst[i]]
        w = np.random.rand(tx.shape[1], 1)
        ty=y[i].reshape(y[i].shape[0],1)

        #print('Training for subset {0}\n'.format(i))
        # start the logistic regression
        for iter in range(max_iter):
            
            # get loss and update w
            loss, w = learning_by_gradient_descent(ty, tx, w, gamma)
            
            # log info
            #if iter % 100 == 0:
                #print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
            
            #store loss
            losses.append(loss)
            
            # converge criterion
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break

        print("Best loss for subset {i}={l}".format(i=i,l=calculate_loss_logistic(ty, tx, w)))
        all_losses.append(losses)
        weights.append(w)
    return weights,all_losses