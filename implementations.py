def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

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

def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for batcy,batcx in batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
   
            stoc_grad = compute_stoch_gradient(batcy,batcx,w)
            loss = compute_loss(y,tx,w)
            w = w-gamma*stoc_grad
    
            ws.append(w)
            losses.append(loss)
    
    return losses[-1], ws[-1]

#Ridge regression

def ridge_regression(y, tx, lambda_):
    aI = 2 * len(y) * lambda_ * np.identity(tx.shape[1])
    A = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(A,b)
    return w

#Least squares with normal equation

def least_squares(y, tx):
    """calculate the least squares solution."""
    A = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)
    e = y - tx.dot(w)
    return w, calculate_mse(e)

#Logistic regression

