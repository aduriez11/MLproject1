import numpy as np

def build_k_indices(y, k_fold, seed):
    """
    Build k indices for k-folds
    :param y: ndarray: predicted values
    :param k_fold: int: number of folds
    :param seed: int: seed for random number generator
    
    :return: tuple(ndarray,ndarray): list of indices for all folds and permutated indexes of predicted values
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices),indices