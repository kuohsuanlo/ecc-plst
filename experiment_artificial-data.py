import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_multilabel_classification
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

def convertFeatureToInteger(table,col_index):
    type_list = []
    for row in table:
        if row[col_index] not in type_list :
            type_list.append(row[col_index])

    for row in table :
        for i in range (np.size(type_list)) :
            if row[col_index] == type_list[i]:
                row[col_index] = i


if __name__ == '__main__':

#   Generating artificial data.
    #n_labels*3<=n_classes
    

    n_samples = 2407
    n_classes=294
    n_labels=6
    '''
    X, Y = make_multilabel_classification(n_samples,n_classes, n_labels,
                                          allow_unlabeled=True,
                                          return_indicator=True,
                                          random_state=1)

    np.random.seed(0)
    order = np.random.permutation(n_samples)
    X = X[order]
    Y = Y[order].astype(np.float)
    
    np.savetxt('X.data', X,fmt='%1.4e') 
    np.savetxt('Y.data', Y,fmt='%1.4e')
    '''
    X = np.loadtxt('X.data')    
    Y = np.loadtxt('Y.data')

    X_train = X[0:n_samples/2]
    Y_train = Y[0:n_samples/2]

    X_test  = X[n_samples/2 : n_samples]
    Y_test  = Y[n_samples/2 : n_samples]
    H       = np.zeros((len(X_test),n_labels))


    #print X
    #print Y
    #print H

#   First try. Focus on the first feature

    y_train = Y_train[:,0]
    y_test  = Y_test[:,0]


    #print X_train
    #print y_train


    clf = OneVsRestClassifier(SVC(kernel='linear'))
    #clf = svm.SVC(kernel='poly', gamma=10)
    clf.fit(X_train, y_train)
    

    error=0;
    for i in range(len(X_test)):
        if clf.predict(X_test[i]) != (y_test[i]) :
            error+=1
    
    print '==== ==== Discard all except 1st feature'    
    print 'error = ',error / (len(X_test)*1.000)
    #Result ~= 0.2


'''
#   Binary Relavance

    #Training N=n_labels model
    for feature in range (n_labels):
        #print feature
        y_train = Y_train[:,feature]
        y_test  = Y_test[:,feature]

        #print X_train
        #print y_train
        #print H
        
        clf = OneVsRestClassifier(SVC(kernel='linear'))
        #clf = svm.SVC(kernel='poly', gamma=10)
        clf.fit(X_train, y_train)
        
        for i in range(len(Y_test)):
            H[i,feature] = clf.predict(X_test[i])         

    #Calculating Diff
    
    print '==== ==== BR'  
    error=0
    featrue_error=0;
    for j in range(n_labels) :
        feature_error=0
        for i in range(len(Y_test)) :
            if H[i,j]!=Y_test[i,j]:
                error+=1
                feature_error+=1;
        #print 'feature_error = ',feature_error / (len(X_test)*1.000)
      
 
    print '0/1 diff = ',error 
    print '0/1 loss = ', error / ((len(X_test)*1.000)*n_labels)
    
#   RAKel





for row in table:
    print row[1:8]



X = table[:5]  # we only take the first two features.
Y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
'''


