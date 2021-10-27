import numpy as np

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def standartize(x):
    """Data standartization
    param x: vector of numbers
    """
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    
    return std_data

def calculate_mse_(y, tx, w):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)

#Gradient descent

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # update w by gradient descent
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)

    return losses[-1], ws[-1]

#Stochastic Gradient Descent

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e = y-tx@w
    stoc_grad = -1/len(y) * np.transpose(tx)@e
    return stoc_grad
    

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
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
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for batcy,batcx in batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
            stoc_grad = compute_stoch_gradient(batcy,batcx,w)
            error=y-tx.dot(w)
            loss = calculate_mse(error)
            w = w-gamma*stoc_grad
            ws.append(w)
            losses.append(loss)
    return losses[-1], ws[-1]

#Least squares with normal equation

def least_squares(y, tx):
    """calculate the least squares solution."""
    A = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)
    e = y - tx.dot(w)
    return w, calculate_mse(e)

#Polynomial regression

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def plot_fitted_curve(y, x, weights, degree, ax):
    """plot the fitted curve."""
    ax.scatter(x, y, color='b', s=12, facecolors='none', edgecolors='r')
    xvals = np.arange(min(x) - 0.1, max(x) + 0.1, 0.1)
    tx = build_poly(xvals, degree)
    f = tx.dot(weights)
    ax.plot(xvals, f)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Polynomial degree " + str(degree))

#Ridge regression

#Ridge regression

def ridge_regression(y, tx, lambda_):
    aI = 2 * len(y) * lambda_ * np.identity(tx.shape[1])
    A = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(A,b)
    return w

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices),indices
    
#Logistic regression

#Cross-validation
def cross_validation_ridge(y, x, k_indices, k, lambda_, degree):
  
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
    
    loss_tr = np.sqrt(2*calculate_mse_(y_train, x_trainpoly, w))
    loss_te= np.sqrt(2*calculate_mse_(y_test, x_testpoly, w))
  
    return loss_tr, loss_te , w


def cross_validation_leastsquares(y, x, k_indices, k, degree):
  
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
    loss_tr = np.sqrt(2*calculate_mse_(y_train, x_trainpoly, w))
    loss_te= np.sqrt(2*calculate_mse_(y_test, x_testpoly, w))
    #print(loss_tr,loss_te)
  
    return loss_tr, loss_te

