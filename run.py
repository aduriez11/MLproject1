import numpy as np
import matplotlib.pyplot as plt
import os
import csv

## Some useful functions

def load_csv_data(data_path, sub_sample=False):
    """
    Loads data from csv file
    :param data_path: string: path to the file
    :param sub_sample: boolean: whether to load sample or not
    
    :return: tuple(ndarray,ndarray,ndarray): y (class labels), tX (features) and ids (event ids)
    """
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def standartize(x):
    """
    Data standartization
    :param x: ndarray: vector
    
    :return: ndarray: standartized vector
    """
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    
    return std_data

def calculate_mse(e):
    """
    Calculate the mse for vector e
    :param e: float: error value
    
    :return: float: mean squared error
    """
    return 1/2*np.mean(e**2)

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

def predict_labels(weights, data):
    """
    Generates class predictions
    :param: ndarray: weights 
    :param: ndarray: data matrix
    
    :return: ndarray: predictions
    """
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    :param ids: ndarray: event ids associated with each prediction
    :param y_pred: ndarray: predicted class labels
    :param name: string: name of .csv output file to be created
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

### Main

## Preprocessing of training data

DATA_TRAIN_PATH = 'train.csv' # TODO: download train data and supply path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

tX0=np.copy(tX[tX[:,22]==0,:])
ids0=np.copy(ids[tX[:,22]==0])
y0 = np.copy(y[tX[:,22]==0])

tX1=np.copy(tX[tX[:,22]==1,:])
ids1=np.copy(ids[tX[:,22]==1])
y1 = np.copy(y[tX[:,22]==1])

tX2=np.copy(tX[tX[:,22]==2,:])
ids2=np.copy(ids[tX[:,22]==2])
y2 =  np.copy(y[tX[:,22]==2])

tX3=np.copy(tX[tX[:,22]==3,:])
ids3=np.copy(ids[tX[:,22]==3])
y3 =  np.copy(y[tX[:,22]==3])

#Deleting columns with clusters
tX0=np.copy(np.delete(tX0,(22),axis=1))
tX1=np.copy(np.delete(tX1,(22),axis=1))
tX2=np.copy(np.delete(tX2,(22),axis=1))
tX3=np.copy(np.delete(tX3,(22),axis=1))

#Deleting columns where all values are same
tX0cl=np.copy(tX0[:,np.invert(np.all(tX0 == tX0[0,:], axis = 0))])
tX1cl=np.copy(tX1[:,np.invert(np.all(tX1 == tX1[0,:], axis = 0))])
tX2cl=np.copy(tX2[:,np.invert(np.all(tX2 == tX2[0,:], axis = 0))])
tX3cl=np.copy(tX3[:,np.invert(np.all(tX3 == tX3[0,:], axis = 0))])

#Don't overwrite global variables
tX0=np.copy(tX0cl)
tX1=np.copy(tX1cl)
tX2=np.copy(tX2cl)
tX3=np.copy(tX3cl)

#Deleting rows with NaNs
ids0d=np.copy(ids0[tX0[:,0]!=-999.0])
y0d = np.copy(y0[tX0[:,0]!=-999.0])
tX0=np.copy(tX0[tX0[:,0]!=-999.0,:])

ids1d=np.copy(ids1[tX1[:,0]!=-999.0])
y1d = np.copy(y1[tX1[:,0]!=-999.0])
tX1=np.copy(tX1[tX1[:,0]!=-999.0,:])

ids2d=np.copy(ids2[tX2[:,0]!=-999.0])
y2d = np.copy(y2[tX2[:,0]!=-999.0])
tX2=np.copy(tX2[tX2[:,0]!=-999.0,:])

ids3d=np.copy(ids3[tX3[:,0]!=-999.0])
y3d = np.copy(y3[tX3[:,0]!=-999.0])
tX3=np.copy(tX3[tX3[:,0]!=-999.0,:])

#Data standartization
tX0st=standartize(tX0)
tX1st=standartize(tX1)
tX2st=standartize(tX2)
tX3st=standartize(tX3)

y = [y0d, y1d, y2d, y3d]
tXst = [tX0st, tX1st, tX2st, tX3st]

## Preprocessing of test data

DATA_TEST_PATH = 'test.csv' 
y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

tX0_test=tX_test[tX_test[:,22]==0,:-1]
tX1_test=tX_test[tX_test[:,22]==1,:]
tX2_test=tX_test[tX_test[:,22]==2,:]
tX3_test=tX_test[tX_test[:,22]==3,:]

ids0_test=ids_test[tX_test[:,22]==0]
ids1_test=ids_test[tX_test[:,22]==1]
ids2_test=ids_test[tX_test[:,22]==2]
ids3_test=ids_test[tX_test[:,22]==3]

tX0_test=np.delete(tX0_test,(22),axis=1)
tX0_test=tX0_test[:,np.count_nonzero(tX0_test==-999.0, axis = 0)!=tX0_test.shape[0]]

tX1_test=np.delete(tX1_test,(22),axis=1)
tX1_test=tX1_test[:,np.count_nonzero(tX1_test==-999.0, axis = 0)!=tX1_test.shape[0]]

tX2_test=np.delete(tX2_test,(22),axis=1)
tX3_test=np.delete(tX3_test,(22),axis=1)

centered_data0 = tX0_test - np.mean(tX0, axis=0)
centered_data1 = tX1_test - np.mean(tX1, axis=0)
centered_data2 = tX2_test - np.mean(tX2, axis=0)
centered_data3 = tX3_test - np.mean(tX3, axis=0)
std0 = np.std(tX0 - np.mean(tX0, axis=0), axis=0)
std1 = np.std(tX1 - np.mean(tX1, axis=0), axis=0)
std2 = np.std(tX2 - np.mean(tX2, axis=0), axis=0)
std3 = np.std(tX3 - np.mean(tX3, axis=0), axis=0)
#Data standartization
tX0st_test=centered_data0 / std0
tX1st_test=centered_data1 / std1
tX2st_test=centered_data2 / std2
tX3st_test=centered_data3 / std3

## Finding the weights

w0 = ridge_regression(y[0], build_poly(tXst[0], 1), 0.01)
w1 = ridge_regression(y[1], build_poly(tXst[1], 5), 0.12328467394420659)
w2 = ridge_regression(y[2], build_poly(tXst[2], 4), 0.01)
w3 = ridge_regression(y[3], build_poly(tXst[3], 4), 0.01)
w_list = [w0, w1, w2, w3]

## Making the predictions

tXst_test = [tX0st_test, tX1st_test, tX2st_test, tX3st_test]
ids_test_list = [ids0_test, ids1_test,ids2_test,ids3_test]

ypred0 = predict_labels(w_list[0], build_poly(tXst_test[0], 1))
ypred1 = predict_labels(w_list[1], build_poly(tXst_test[1], 5))
ypred2 = predict_labels(w_list[2], build_poly(tXst_test[2], 4))
ypred3 = predict_labels(w_list[3], build_poly(tXst_test[3], 4))
ypred_list = [ypred0, ypred1, ypred2, ypred3]

## Making the submission

ypred_1 = np.hstack((ypred_list[0], ypred_list[1]))
ypred_2 = np.hstack((ypred_list[2], ypred_list[3]))
ypred = np.hstack((ypred_1, ypred_2))
ids_test1 = np.hstack((ids_test_list[0], ids_test_list[1]))
ids_test2 = np.hstack((ids_test_list[2], ids_test_list[3]))
ids_test = np.hstack((ids_test1, ids_test2))

# Sorting the ids

pred_set = np.column_stack((ids_test, ypred))
pred_set_sorted = pred_set[pred_set[:, 0].argsort()]
ids_test_sorted = pred_set_sorted[:,0]
ypred_sorted = pred_set_sorted[:,1]

create_csv_submission(ids_test_sorted, ypred_sorted, "ridge_regression.csv")

