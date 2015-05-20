import numpy as np
import matplotlib.pyplot as plt
import pprint
import random

from sklearn.datasets import make_multilabel_classification
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

def convertFeatureToInteger(table,col_index):
    type_list = []
    for row in table:
        if row[col_index] not in type_list :
            type_list.append(row[col_index])

    for row in table :
        for i in range (np.size(type_list)) :
            if row[col_index] == type_list[i]:
                row[col_index] = i

def mostCommon(lst):
    return max(set(lst), key=lst.count)

def generateData(n_samples,n_classes,n_labels):
    X, Y = make_multilabel_classification(n_samples,n_classes, n_labels,
                                          allow_unlabeled=True,
                                          return_indicator=True,
                                          random_state=1)

    np.random.seed(0)
    order = np.random.permutation(n_samples)
    X = X[order]
    Y = Y[order].astype(np.float)
    
    np.savetxt('datasets/af-multilabel/X.data', X,fmt='%1.4e') 
    np.savetxt('datasets/af-multilabel/Y.data', Y,fmt='%1.4e')

def firstFeature(X,Y):
#   First try. Focus on the first feature
    
    X_train = X[0:n_samples/2]
    Y_train = Y[0:n_samples/2]

    X_test  = X[n_samples/2 : n_samples]
    Y_test  = Y[n_samples/2 : n_samples]
    
    y_train = Y_train[:,0]
    y_test  = Y_test[:,0]


    #print X_train
    #print y_train


    #clf = OneVsRestClassifier(SVC(kernel='linear'))
    #clf = svm.SVC(kernel='poly', gamma=10)
    clf  = AdaBoostClassifier()
    clf.fit(X_train, y_train)
    
    
    #Getting the predicted answer
    H = clf.predict(X_test)

    #Calculating error
    error=0;
    for i in range(len(y_test)):
        if  H[i] != y_test[i] :
            error+=1
    
    #print H
    #print y_test  
    print '==== ==== 1st feature'    
    print 'error = ',error / (len(X_test)*1.000)
    #Result ~= 0.2
def binaryRelavance(X,Y):

    #Initialize data

    X_train = X[0:n_samples/2]
    Y_train = Y[0:n_samples/2]

    X_test  = X[n_samples/2 : n_samples]
    Y_test  = Y[n_samples/2 : n_samples]

    H       = np.zeros((len(X_test),n_labels))

    #Training N=n_labels model
    for feature in range (n_labels):
        #print feature
        y_train = Y_train[:,feature]
        y_test  = Y_test[:,feature]

        #print X_train
        #print y_train
        #print H
        
        #clf = OneVsRestClassifier(SVC(kernel='linear'))
        #clf = svm.SVC(kernel='poly', gamma=10)
        clf  = AdaBoostClassifier()
        clf.fit(X_train, y_train)
        
        #Getting the predicted answer
        H[:,feature] = clf.predict(X_test)

    #print H
    #print Y_test

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
        print 'feature_error = ',feature_error / (len(X_test)*1.000)
      
    #print '0/1 diff = ',error 
    print '0/1 loss = ', error / ((len(X_test)*1.000)*n_labels)


def rakel(X,Y):
    n_labeltuples = 15
    n_labelk = 3

    #Initial tuple's elements to (-1,-1...,-1)
    labelTuples = np.zeros((n_labeltuples,n_labelk))
    labelTuples= labelTuples-1
    #print labelTuples


    #Construct n_labeltuples subset , each containing n_labelk elements
    for i in range(n_labeltuples):
        #print i
        j=0    
        while True:
            new_id = random.choice(range(n_labels)) #0~n_labels-1
            if new_id not in labelTuples[i]:
                labelTuples[i][j]=(new_id)
                j+=1
            if j is 3:
                break
    #print labelTuples



    #For each tuple, do a powerset algorithms with BR



if __name__ == '__main__':

    #Generating artificial data.
    #n_labels*3<=n_classes
    n_samples = 2417
    n_classes=103
    n_labels=14

    #generateData(n_samples,n_classes,n_labels);
    
    #Loading Data    
    X = np.loadtxt('datasets/af-multilabel/X.data')    
    Y = np.loadtxt('datasets/af-multilabel/Y.data')

    #X_train = X[0:n_samples/2]
    #Y_train = Y[0:n_samples/2]

    #X_test  = X[n_samples/2 : n_samples]
    #Y_test  = Y[n_samples/2 : n_samples]
   

    #print X
    #print Y
    #print H



#   First try. Focus on the first feature
    #firstFeature(X,Y)


#   Binary Relavance
    #binaryRelavance(X,Y)


#   RAKel
    rakel(X,Y)




'''
H       = np.zeros((len(X_test),n_labels))
    

    #Training N=n_labels model
    for feature in range (n_labels):
        #print feature
        y_train = Y_train[:,feature]
        y_test  = Y_test[:,feature]

        #print X_train
        #print y_train
        #print H
        
        #clf = OneVsRestClassifier(SVC(kernel='linear'))
        #clf = svm.SVC(kernel='poly', gamma=10)
        clf  = AdaBoostClassifier()
        clf.fit(X_train, y_train)
        
        #Getting the predicted answer
        H[:,feature] = clf.predict(X_test)

    #print H
    #print Y_test

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
        print 'feature_error = ',feature_error / (len(X_test)*1.000)
      
    #print '0/1 diff = ',error 
    print '0/1 loss = ', error / ((len(X_test)*1.000)*n_labels)


'''


