import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

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

    #   Preprocessing the data into numerical X,Y


    yeast = open('../dataset/yeast/yeast.data','r')
    table = [row.strip().split() for row in yeast]


    #   Converting first attribute from text to interger.
    convertFeatureToInteger(table,0)
    convertFeatureToInteger(table,9)

    X = np.array(table)[:,0:8]
    y = np.array(table)[:,9]


    print X
    print y



'''
for row in table:
    print row[1:8]



X = table[:5]  # we only take the first two features.
Y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
'''


