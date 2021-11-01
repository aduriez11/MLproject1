import numpy as np

def standartize(x):
    """
    Data standartization
    :param x: ndarray: vector
    
    :return: ndarray: standartized vector
    """
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    return std_data
    

def general_cleaning (y, tX, ids):
    """
    Initial cleaning of data (splitting to different subsets, droping columns with the same value)
    :param y: ndarray: predicted values
    :param tX: ndarray: regressors
    :param ids: ndarray: identificators of rows
    
    :return: tuple(list[ndarray],list[ndarray],list[ndarray]) lists of predicted values, regressors and identificators for all the subsets
    """
    #splitting to four subsets by integer column
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
    
    return [y0,y1,y2,y3],[tX0cl,tX1cl,tX2cl,tX3cl], [ids0,ids1,ids2,ids3]

def first_column_nan_deleting(y,tX,ids):
    """
    Delete all rows with NaNs (-999.0) in first column
    :param y: list[ndarray]: list of predicted values for all subsets
    :param tX: list[ndarray]: list of regressors for all subsets
    :param ids: list[ndarray]: list of identificators of rows for all subsets
    
    :return: tuple(list[ndarray],list[ndarray],list[ndarray]) lists of predicted values, regressors and identificators for all the subsets after deleting NaNs
    """
    y_out=[]
    tX_out=[]
    ids_out=[]
    for i in range(len(y)):
        #Deleting rows with NaNs
        ids_out.append(np.copy(ids[i][tX[i][:,0]!=-999.0]))
        y_out.append(np.copy(y[i][tX[i][:,0]!=-999.0]))
        tX_out.append(np.copy(tX[i][tX[i][:,0]!=-999.0,:]))
    
    return y_out,tX_out,ids_out

def first_column_nan_deleting_test(tX,ids):
    """
    Delete all rows with NaNs (-999.0) in first column
    :param tX: list[ndarray]: list of regressors for all subsets
    :param ids: list[ndarray]: list of identificators of rows for all subsets
    
    :return: tuple(list[ndarray],list[ndarray]) lists of regressors and identificators for all the subsets after deleting NaNs
    """
    tX_out=[]
    ids_out=[]
    for i in range(len(tX)):
        #Deleting rows with NaNs
        ids_out.append(np.copy(ids[i][tX[i][:,0]!=-999.0]))
        tX_out.append(np.copy(tX[i][tX[i][:,0]!=-999.0,:]))
    
    return tX_out,ids_out

def first_column_nan_substitute(tX):
    """
    Substitute NaNs (-999.0) in the first column with mean
    :param tX: list[ndarray]: regressors
    
    :return: list[ndarray] lists of regressors for all the subsets after replacing NaNs
    """   
    tX_out=[]
    for i in range(len(tX)):
        #Calculate mean for 1st column without -999.0 values
        mean=tX[i][tX[i][:,0]!=-999.0].mean(axis=0)[0]
        
        #Substitute values -999.0 in the first column with mean of the column
        tX_out.append(np.copy(np.where(tX[i]==-999.0,mean,tX[i])))

    return tX_out

def drop_correlated_columns_nan_deleted(tX):
    """
    Drop columns among regressors with strong correlation
    :param tX: list[ndarray]: list of regressors for each subset
    
    :return: list[ndarray] lists of regressors for all the subsets after deleting columns
    """ 
    tX0=tX[0]
    tX1=tX[1]
    tX2=tX[2]
    tX3=tX[3]
    
    #Droping correlated columns after nan deling
    tX0=tX0[:,[i for i in range(tX0.shape[1]) if i not in [0,3,6]]]
    tX1=tX1[:,[i for i in range(tX1.shape[1]) if i not in [0,3,6,21]]]
    tX2=tX2[:,[i for i in range(tX2.shape[1]) if i not in [0,3,4,9,22]]]
    tX3=tX3[:,[i for i in range(tX3.shape[1]) if i not in [0,9,21,22]]]
    
    return [tX0,tX1,tX2,tX3]

def drop_correlated_columns_nan_substituted(tX):
    """
    Drop columns among regressors with strong correlation
    :param tX: list[ndarray]: list of regressors for each subset
    
    :return: list[ndarray] lists of regressors for all the subsets after deleting columns
    """
    tX0=tX[0]
    tX1=tX[1]
    tX2=tX[2]
    tX3=tX[3]
    
    #Droping correlated columns after nan deling
    tX0=tX0[:,[i for i in range(tX0.shape[1]) if i not in [0,3,6]]]
    tX1=tX1[:,[i for i in range(tX1.shape[1]) if i not in [0,3,6,21]]]
    tX2=tX2[:,[i for i in range(tX2.shape[1]) if i not in [0,3,4,9,22]]]
    tX3=tX3[:,[i for i in range(tX3.shape[1]) if i not in [0,9,21,22]]]
    
    return [tX0,tX1,tX2,tX3]

def standartization(tX):
    """
    Standartize all the subsets
    :param tX: list[ndarray]: list of regressors for each subset
    
    :return: list[ndarray] lists of regressors for all the subsets after standartization
    """
    tX_out=[]
    for i in range(len(tX)):
        tX_out.append(standartize(tX[i]))
                      
    return tX_out