

import numpy as np
import matplotlib.pyplot as plt
import pprint
import random
import itertools as itool
import sys

from time import sleep

from operator import mul    # or mul=lambda x,y:x*y
from fractions import Fraction

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import datasets
from sklearn.datasets import make_multilabel_classification
from sklearn import svm
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from itertools import chain, combinations


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
    
    np.savetxt('datasets/af-multilabel/X.data', X,fmt='%d') 
    np.savetxt('datasets/af-multilabel/Y.data', Y,fmt='%d')

def generatePowerset(labelTuple,n_labelk):
    #R = []
    #for labelTuple in labelTuples:
    i = set(labelTuple)
    A=[]        
    for z in chain.from_iterable(combinations(i, r) for r in range(len(i)+1)):              
        z = np.array(z)
        zeros = (np.zeros((1,n_labelk-len(z))))[0]-1  
        newrow =np.concatenate((z,zeros),axis=0)
        newrow = sorted(newrow)
        A.append(newrow)       
       #R.append( A)
    return np.array(A).astype(int)

#In reversed order, for example 8 = 0001
def bin(i, n_labelk):
    if i == 0:
        return "0"*n_labelk
    s = ''
    while i:
        if i & 1 == 1:
            s = s+"1" 
        if i & 1 != 1:
            s = s+"0" 
        i >>= 1
    s+= "0"*(n_labelk-len(s))
    return s



def nCk(n,k): 
    return int( reduce(mul, (Fraction(n-i, i+1) for i in range(k)), 1) )

def firstFeature(X,Y):
    print '==== ==== 1st feature'   
    X_train = X[0:n_samples/2]
    Y_train = Y[0:n_samples/2]

    X_test  = X[n_samples/2 : n_samples]
    Y_test  = Y[n_samples/2 : n_samples]
    
    y_train = Y_train[:,0]
    y_test  = Y_test[:,0]


    #print X_train
    #print y_train


    #clf = OneVsRestClassifier(LinearSVC(random_state=0))
    #clf  = OneVsOneClassifier(LinearSVC(random_state=0))
    #clf = svm.SVC(kernel='poly', gamma=10)
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
 
    print 'error = ',error / (len(X_test)*1.000)
    #Result ~= 0.2
def binaryRelavance(X,Y):
    print '==== ==== BR'  
    #Initialize data
    X_train = X[0:n_samples/2]
    Y_train = Y[0:n_samples/2]

    X_test  = X[n_samples/2 : n_samples]
    Y_test  = Y[n_samples/2 : n_samples]

    H       = np.zeros((len(X_test),n_labels))

    print 'Total : ',n_labels,' labels'
    #Training N=n_labels modeld
    for feature in range (n_labels):
        #Progress bar
        sys.stdout.write('\r')
        i = int(round((20*(0.01+feature))/(n_labels)))
        sys.stdout.write("[%-20s] %d%%" % ('='*i, 5*i))
        sys.stdout.flush()


        #print feature
        #print 'BinaryRalevance Training : ',feature,'/',(n_labels)
        y_train = Y_train[:,feature]
        y_test  = Y_test[:,feature]

        #print X_train
        #print y_train
        #print H
        
         
        
        #clf  = OneVsOneClassifier(LinearSVC(random_state=0))
        #clf  = AdaBoostClassifier()
        clf.fit(X_train, y_train)
        
        #Getting the predicted answer
        H[:,feature] = clf.predict(X_test)

    #print H
    #print Y_test

    #Calculating Diff

    error=0
    featrue_error=0;
    for j in range(n_labels) :
        feature_error=0
        for i in range(len(Y_test)) :
            if H[i,j]!=Y_test[i,j]:
                error+=1
                feature_error+=1;
        #print 'feature_error = ',feature_error / (len(X_test)*1.000)
      
    #print '0/1 diff = ',error 
    print '\n0/1 loss = ', error / ((len(X_test)*1.000)*n_labels)
    #~0.108

def rakel(X,Y):
    print '==== ==== Rakel'  
    n_labelk = 3
    n_labeltuples = 100
    n_labeltuples = min(nCk(n_labels,n_labelk),n_labeltuples)  # Rakel_o
    #Initialize data

    X_train = X[0:n_samples/2]
    X_test  = X[n_samples/2 : n_samples]
    
    #Y_train = Y[0:n_samples/2]
    Y_test  = Y[n_samples/2 : n_samples]

    #Initialize the tuples' elements to (-1,-1...,-1)
    labelTuples = []

    #Construct n_labeltuples subset , each containing n_labelk elements
    iters = list(itool.combinations(np.arange(n_labels),n_labelk))
    while True:        
        new_set = random.choice(iters) #0~n_labels-1
        if new_set not in labelTuples:
            labelTuples.append(new_set)
        if len(labelTuples) == n_labeltuples:
            break 
    
    #Generating sorted k features tuples from C(n_labelk, n_labels)'s powerset R = in binary form,  -1= ith feature ==0, >=0 = ith feature ==1  
    Y_t= []  # Y labelspace truth table
    for labelTuple in labelTuples:
        labelTuple = sorted(labelTuple)
        #print labelTuple
        R = generatePowerset(labelTuple,n_labelk)
        Y_t_r = []
        for subset in R:
            tmp = labelTuple[:]
            for i,label in enumerate(labelTuple) :
                if label in subset:
                    tmp[i] = tuple([label, 1])
                else:
                    tmp[i] = tuple([label,-1])
            subset = tmp
            Y_t_r.append( subset)
        Y_t.append(Y_t_r)
    Y_t = np.array(Y_t).astype(int) 
    
    #Genarate transformed Y based on  Y_t table
    #n_noStamps  = np.zeros((len(X_test),n_labels))
    n_yesStamps = np.zeros((len(X_test),n_labels))
    n_allStamps = np.zeros((len(X_test),n_labels))


    print 'Total : ',n_labeltuples,' hypothesis'
    for hi,sets in enumerate(Y_t): # loop n_labeltuples times
        #Progress bar
        sys.stdout.write('\r')
        i = int(round((20*(0.01+hi))/(len(Y_t))))
        sys.stdout.write("[%-20s] %d%%" % ('='*i, 5*i))
        sys.stdout.flush()
       
        #box = np.zeros((len(X_test),n_labels))-1
        stuple = sorted(labelTuples[hi])
        #print 'In k label sets :',stuple
        #print 'Rakel labelPSet Training : ',hi,'/',len(Y_t)
        ted_YN=[]
        bted_YN=[]
        transformedY = []
        binarytransformedY = []
        for row in Y:
            trandsfomedY_row =[]
            for iset in sets:
                #print iset
                #print '--------'
                n_match =0
                for i in iset:
                    if i[1]== 1  and  row[i[0]]== 1:
                        #same
                        n_match+=1
                    if i[1]==-1  and  row[i[0]]== 0:
                        n_match+=1
                
                if n_match == n_labelk:
                    trandsfomedY_row.append(1)
                else:
                    trandsfomedY_row.append(0)
            #Binary class to numeric indicator
            binarytransformedY.append(trandsfomedY_row)
            for li, label in enumerate(trandsfomedY_row):
                if label != 0:
                    transformedY.append(li)
            
        transformedY= np.matrix(transformedY)
        
        
        #Generate tranformed Y_0~ Y_labeltuples based on Y_t table
        filename = 'datasets/af-multilabel/processed/Rakel/Y_'+repr( hi)+'.data'
        np.savetxt(filename, transformedY,fmt='%d')
        ted_Y = np.loadtxt(filename)
        
        #Generate tranformed Y_0~ Y_labeltuples in binaryform

        #print ted_Y
        Y_train = ted_Y[0:n_samples/2]
        
        #clf = OneVsRestClassifier(LinearSVC(random_state=0))
        #clf  = OneVsOneClassifier(LinearSVC(random_state=0))
        clf.fit(X_train, Y_train)
        
        #Getting the predicted answer
        H= clf.predict(X_test)
        
        #Get one's election box
        for i_test,i in enumerate(H.astype(int)):
            s=bin(i, n_labelk)
            for i_ind,indicator in enumerate(s):
                if indicator =='1' :
                    #Voting
                    n_yesStamps[i_test,stuple[i_ind]]+=1
                    n_allStamps[i_test,stuple[i_ind]]+=1
                    #box[i_test,stuple[i_ind]]=1
                if indicator =='0':
                    #n_noStamps[i_test,stuple[i_ind]]+=1
                    n_allStamps[i_test,stuple[i_ind]]+=1
                    #box[i_test,stuple[i_ind]]=0

    #Start election  
    electedH= np.zeros((len(X_test),n_labels))    
    for i in range(len(X_test)):
        for j in range(n_labels):
            if n_allStamps[i][j]==0:
                electedH[i][j]=0
            else:
                electedH[i][j] = round( n_yesStamps[i][j] / n_allStamps[i][j]*(1.0) )
    
    #Open the election box
    
    #Calculating Diff
    error=0
    for j in range(n_labels) :
        for i in range(len(Y_test)) :
            if electedH[i,j]!=Y_test[i,j]:
                error+=1
                
    print '\n0/1 loss = ', error / (len(Y_test)*n_labels*1.000)

def labelPowerset(X,Y):
    print '==== ==== LabelPowerset'  
    #Initialize data

    X_train = X[0:n_samples/2]
    X_test  = X[n_samples/2 : n_samples]
    
    Y_train = Y[0:n_samples/2]
    Y_test  = Y[n_samples/2 : n_samples]

    #Initialize the tuples' elements to (-1,-1...,-1)
    labelTuples = []

    #Construct n_labeltuples subset , each containing n_labelk elements
    labelTuples.append(np.arange(n_labels))
    
    
    Y_t= []  # Y labelspace truth table
    for labelTuple in labelTuples:
        labelTuple = sorted(labelTuple)
        #print labelTuple
        R = generatePowerset(labelTuple,n_labels)
        Y_t_r = []
        for subset in R:
            tmp = labelTuple[:]
            for i,label in enumerate(labelTuple) :
                if label in subset:
                    tmp[i] = tuple([label, 1])
                else:
                    tmp[i] = tuple([label,-1])
            subset = tmp
            Y_t_r.append( subset)
        Y_t.append(Y_t_r)
    Y_t = np.array(Y_t).astype(int)[0]
   
    #Genarate transformed Y based on  Y_t table

    ted_YN=[]
    transformedY = []
    binarytransformedY = []
    
       
    for row in Y:
        trandsfomedY_row =[]
        for iset in Y_t:
            #print iset
            #print '--------'
            n_match =0
            for i in iset:
                if i[1]==  1  and  row[i[0]]== 1:
                    #same
                    n_match+=1
                if i[1]== -1  and  row[i[0]]== 0:
                    n_match+=1
            if n_match == n_labels:
                trandsfomedY_row.append(1)
            else:
                trandsfomedY_row.append(0)
        #Binary class to numeric indicator
        binarytransformedY.append(trandsfomedY_row)
        for li, label in enumerate(trandsfomedY_row):
            if label != 0:
                transformedY.append(li)
        
    transformedY= np.matrix(transformedY)
     
    filename = 'datasets/af-multilabel/processed/LabelPowerSet/Y_LP.data'
    np.savetxt(filename, transformedY,fmt='%d')
    ted_Y = np.loadtxt(filename)
    

    #print ted_Y
    Y_train = ted_Y[0:n_samples/2]
    
    #clf = OneVsRestClassifier(LinearSVC(random_state=0))
    #clf  = OneVsOneClassifier(LinearSVC(random_state=0))
    clf.fit(X_train, Y_train)
    
    #Getting the predicted answer
    H= clf.predict(X_test)
   

    #Transform to binary indicator
    H_in_binary = np.zeros((n_samples/2,n_labels))
    for i_test,i in enumerate(H.astype(int)):
        s=bin(i, n_labels)
        for i_ind,indicator in enumerate(s):
            if indicator =='1' :
                H_in_binary[i_test,i_ind]+=1
               
    #Calculating Diff
    error=0
    for j in range(n_labels) :
        for i in range(len(Y_test)) :
            if H_in_binary[i,j]!=Y_test[i,j]:
                error+=1

    print '\n0/1 loss = ', error / (len(Y_test)*n_labels*1.000)


if __name__ == '__main__':

    #Generating artificial data.
    #n_labels*3<=n_classes
    n_samples = 10000
    n_classes=24
    n_labels=6

    generateData(n_samples,n_classes,n_labels);
    
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

#   Select Base learner
    clf  = OneVsOneClassifier(LinearSVC(random_state=0,max_iter=5000))

#   First try. Focus on the first feature
    firstFeature(X,Y)


#   Binary Relavance
    binaryRelavance(X,Y)


#   RAKel
    rakel(X,Y)

#   LabelPowerset
    labelPowerset(X,Y)
