# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:11:16 2020

@author: Amina Asif

Modified by Muhammad Waleed Usman
i19-2140 - MSDS
"""

# importing the libraries
from scipy import spatial
from numpy.random import randn,randint #importing randn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np #importing numpy
import matplotlib.pyplot as plt #importing plotting module
import itertools
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import kde
import timeit

def plotDensity_2d(X,Y):
    nbins = 200
    minx, maxx = np.min(X[:,0]), np.max(X[:,0])
    miny, maxy = np.min(X[:,1]), np.max(X[:,1])
    xi, yi = np.mgrid[minx:maxx:nbins*1j, miny:maxy:nbins*1j]
    def calcDensity(xx):
        k = kde.gaussian_kde(xx.T)        
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        return zi.reshape(xi.shape)
    pz=calcDensity(X[Y==1,:])
    nz=calcDensity(X[Y==-1,:])
    
    c1=plt.contour(xi, yi, pz,cmap=plt.cm.Greys_r,levels=np.percentile(pz,[75,90,95,97,99])); plt.clabel(c1, inline=1)
    c2=plt.contour(xi, yi, nz,cmap=plt.cm.Purples_r,levels=np.percentile(nz,[75,90,95,97,99])); plt.clabel(c2, inline=1)
    plt.pcolormesh(xi, yi, 1-pz*nz,cmap=plt.cm.Blues,vmax=1,vmin=0.99);plt.colorbar()
    markers = ('s','o')
    plt.scatter(X[Y==1,0],X[Y==1,1],marker = markers[0], c = 'y', s = 30)
    plt.scatter(X[Y==-1,0],X[Y==-1,1],marker = markers[1],c = 'c', s = 30)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')   
    plt.grid()
    plt.show()
                   

def plotit(X,Y=None,clf=None, markers = ('s','o'), hold = False, transform = None):
    """
    Just a function for showing a data scatter plot and classification boundary
    of a classifier clf
    """
    eps=1e-6
    minx, maxx = np.min(X[:,0]), np.max(X[:,0])
    miny, maxy = np.min(X[:,1]), np.max(X[:,1])
    
    if clf is not None:
        npts = 150
        x = np.linspace(minx,maxx,npts)
        y = np.linspace(miny,maxy,npts)
        t = np.array(list(itertools.product(x,y)))
        if transform is not None:
            t = transform(t)
        z = clf(t)
        z = np.reshape(z,(npts,npts)).T        
        extent = [minx,maxx,miny,maxy]
        plt.contour(x,y,z,[-1+eps,0,1-eps],linewidths = [2],colors=('b','k','r'),extent=extent, label='f(x)=0')
        #plt.imshow(np.flipud(z), extent = extent, cmap=plt.cm.Purples, vmin = -2, vmax = +2); plt.colorbar()
        plt.pcolormesh(x, y, z,cmap=plt.cm.Purples,vmin=-2,vmax=+2);plt.colorbar()
        plt.axis([minx,maxx,miny,maxy])
    
    if Y is not None:
        
        plt.scatter(X[Y==1,0],X[Y==1,1],marker = markers[0], c = 'y', s = 30)
        plt.scatter(X[Y==-1,0],X[Y==-1,1],marker = markers[1],c = 'c', s = 30)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')        
         
    else:
        plt.scatter(X[:,0],X[:,1],marker = '.', c = 'k', s = 5)
    if not hold:
        plt.grid()
        
        plt.show()


def accuracy(ytarget,ypredicted):
    return np.sum(ytarget == ypredicted)/len(ytarget)


class NN:
    def __init__(self):
        pass
    def fit(self, X, Y):
        self.Xtr=X
        self.Ytr=Y
    def predict(self, Xts, k):
        Yts=[]
        Xts=np.array(Xts)
        for t in self.Xtr:
            distances=np.sqrt(np.sum(np.power((Xts-t), 2), axis=1))
            dist = list(zip(self.Xtr,self.Ytr,distances.tolist())) # List of training data with labels and their distances
            dist.sort(key=lambda tup: tup[2]) # sorting the distances using the distance metric
            neighbors = list()
            for i in range(k): # Findind the k neighbors using the least k distances 
                neighbors.append(dist[i][1])
            prediction = max(set(neighbors), key=neighbors.count) # finding the maximum occuring label in the neighbors 
            Yts.append(prediction)
        return Yts


def getExamples(n=100,d=2):
    """
    Generates n d-dimensional normally distributed examples of each class        
    The mean of the positive class is [1] and for the negative class it is [-1]
    DO NOT CHANGE THIS FUNCTION
    """
    Xp = randn(n,d)+1   #generate n examples of the positie class
    #Xp[:,0]=Xp[:,0]+1
    Xn = randn(n,d)-1   #generate n examples of the negative class
    #Xn[:,0]=Xn[:,0]-1
    X = np.vstack((Xp,Xn))  #Stack the examples together to a single matrix
    Y = np.array([+1]*n+[-1]*n) #Associate Labels
    return (X,Y) 


# Function for the prediction using the sklean
def prediction_using_sklearn(k,Xtr,Ytr,Xts,Yts):
    """
    INPUT: K,Testing,Training Arrays
    OUTPUT: Accuracy
    """
    knn = KNeighborsClassifier(n_neighbors=k) # model for knn classifier 
    knn.fit(Xtr,Ytr) # fitting the training data in our model 
    accuracy = knn.score(Xts,Yts) # finding the accuracy for our model 
    return accuracy



# Question no.1 Evaluating Our own classifier and Sklearn
def question1(k,Xtr,Ytr,Xtt,Ytt):
    """
    INPUT: K,Testing,Training Arrays
    OUTPUT: time taken by own classifier, sklearn time 
    """
    start_raw = timeit.default_timer() 
    clf = NN()
    clf.fit(Xtr,Ytr)
    Y = clf.predict(Xtt,k) 
    E = accuracy(Ytt,Y)
    stop_raw = timeit.default_timer()
    time_raw = stop_raw - start_raw
    print('Time taken by own classifier: ', time_raw)
    print("AccuracyRawKnn: ", E)
    start_sklearn = timeit.default_timer()
    accuracysklearn = prediction_using_sklearn(k=k,Xtr=Xtr,Ytr=Ytr,Xts=Xtt,Yts=Ytt)
    stop_sklearn = timeit.default_timer()
    time_sklearn = stop_sklearn - start_sklearn
    print('\n\nTime taken by sklearn classifier: ', time_sklearn)
    print("AccuracySklearnMethod", accuracysklearn)
    if time_raw > time_sklearn:
        print("\nThe performance of sklearn is good as compared to that of our our classifier")
    else:
        print("\nThe performance of sklearn is bad as compared to that of our classifier")
    return time_raw,time_sklearn


# Question no.2 part1 Evaluating Our own classifier and Sklearn
def question2_part1(k,d):
    """
    INPUT: K, dimensions 
    """
    num_training = []
    time_array_raw = []
    time_array_sklearn = []
    for num_of_training in range(10,100,10):
        Xtt,Ytt = getExamples(n=num_of_training,d=d) #Generate Testing Examples
        Xtr,Ytr = getExamples(n=num_of_training,d=d) #Generate Training Examples    
        print("\nFor value of Training examples "+ str(num_of_training)+":")
        raw, sklearnn = question1(k=k,Xtr=Xtr,Ytr=Ytr,Xtt=Xtt,Ytt=Ytt)
        time_array_raw.append(raw)
        time_array_sklearn.append(sklearnn)
        num_training.append(num_of_training)
        print("---------------------------------------")
    plt.plot(num_training,time_array_raw,'or-',time_array_sklearn,'ob-')
    plt.title('Run-time Vs Number of Training examples')
    plt.ylabel('Run time')
    plt.xlabel('Increasing the number of Training examples')
    plt.text(0.5, 0.5, r' blue=sklearn  red=our own classifier')
    plt.grid(True)
    plt.show()


# Question no.2 part2 Evaluating Our own classifier and Sklearn
def question2_part2(k,num_of_testing,num_of_training):
    """
    INPUT: K, number of testing, number of training  
    """
    d_array = []
    time_array_raw = []
    time_array_sklearn = []

    for d in range(1,5):
        Xtt,Ytt = getExamples(n=num_of_training,d=d) #Generate Testing Examples
        Xtr,Ytr = getExamples(n=num_of_training,d=d) #Generate Training Examples    
        print("\nFor value of dimension "+ str(d)+":")
        raw, sklearnn = question1(k=k,Xtr=Xtr,Ytr=Ytr,Xtt=Xtt,Ytt=Ytt)
        time_array_raw.append(raw)
        time_array_sklearn.append(sklearnn)
        d_array.append(d)
        print("---------------------------------------")
    plt.plot(d_array,time_array_raw,'or-',time_array_sklearn,'ob-')
    plt.title('Run-time Vs Number of dimensions')
    plt.ylabel('Run time')
    plt.xlabel('Increasing the number of dimensions')
    plt.text(2, 1, r' blue=sklearn  red=our own classifier')
    plt.grid(True)
    plt.show()

# Question no.2 part3 Evaluating training accuracy with value of k change
def question2_part3(Xtr,Ytr,Xtt,Ytt):
    """
    INPUT: ,Testing,Training Arrays 
    """
    clf = NN()
    clf.fit(Xtr,Ytr)
    k_array = []
    acc_array_own = []
    acc_array_sklearn = []
    for k in range(1,40,2):
        Y = clf.predict(Xtr,k) 
        E = accuracy(Ytr,Y)
        k_array.append(k)
        print("For K of value: ", k, ":")
        print("AccuracyRawKnn: ", E)
        accuracysklearn = prediction_using_sklearn(k=k,Xtr=Xtr,Ytr=Ytr,Xts=Xtr,Yts=Ytr)
        print("\nAccuracySklearnMethod", accuracysklearn)
        acc_array_own.append(E)
        acc_array_sklearn.append(accuracysklearn)
        print("---------------------------------------")
    plt.plot(k_array,acc_array_own,'or-',acc_array_sklearn,'ob-')
    plt.title('Value of k Vs Training Accuracy')
    plt.ylabel('Training Accuracy')
    plt.xlabel('Value of K')
    plt.text(0.05, 0.05, r' blue=sklearn  red=our own classifier')
    plt.grid(True)
    plt.show()

# Question no.2 part4 Evaluating testing accuracy with value of k change
def question2_part4(Xtr,Ytr,Xtt,Ytt):
    """
    INPUT: ,Testing,Training Arrays 
    """
    clf = NN()
    clf.fit(Xtr,Ytr)
    k_array = []
    acc_array_own = []
    acc_array_sklearn = []
    for k in range(1,60,2):
        Y = clf.predict(Xtt,k) 
        E = accuracy(Ytt,Y)
        k_array.append(k)
        print("For K of value: ", k, ":")
        print("AccuracyRawKnn: ", E)
        accuracysklearn = prediction_using_sklearn(k=k,Xtr=Xtr,Ytr=Ytr,Xts=Xtt,Yts=Ytt)
        print("AccuracySklearnMethod", accuracysklearn)
        acc_array_own.append(E)
        acc_array_sklearn.append(accuracysklearn)
        print("---------------------------------------")
    plt.plot(k_array,acc_array_own,'or-',acc_array_sklearn,'ob-')
    plt.title('Value of k Vs Testing Accuracy')
    plt.ylabel('Testing Accuracy')
    plt.xlabel('Value of K')
    plt.text(5, 0.9, r' blue=sklearn  red=our own classifier')
    plt.grid(True)
    plt.show()



    
if __name__ == '__main__':
    #% Data Generation and Density Plotting
    num_of_training = 100 #number of examples of each class
    num_of_testing = 100
    d = 2 #number of dimensions
    k = 3 #Number of k 
    Xtr,Ytr = getExamples(n=num_of_training,d=d) #Generate Training Examples    
    Xtt,Ytt = getExamples(n=num_of_testing,d=d) #Generate Testing Examples 
    print("Number of positive examples in training: ", np.sum(Ytr==1))
    print("Number of negative examples in training: ", np.sum(Ytr==-1))
    print("Dimensions of the data: ", Xtr.shape[1])    
    print("---------------------------------------")
    question1(k=k,Xtr=Xtr,Ytr=Ytr,Xtt=Xtt,Ytt=Ytt) # Question 1 
    question2_part1(k=k,d=d) # Question 2 part a
    question2_part2(k,num_of_testing=num_of_testing,num_of_training=num_of_training) # Question 2 part b
    question2_part3(Xtr=Xtr,Ytr=Ytr,Xtt=Xtt,Ytt=Ytt) # Question 2 part c
    question2_part4(Xtr=Xtr,Ytr=Ytr,Xtt=Xtt,Ytt=Ytt) # Question 2 part d    
    print("---------------------------------------")
    plt.figure()
    plotDensity_2d(Xtr,Ytr)
    plt.title("Train Data")
    plt.figure()
    plotDensity_2d(Xtt,Ytt)
    plt.title("Test Data")
    print("*"*10+"1- Nearest Neighbor Implementation"+"*"*10)
    voronoi_plot_2d(Voronoi(Xtr),show_vertices=False,show_points=False,line_colors='orange')
    plotit(Xtr,Ytr,clf=clf.predict)
    plt.title("K-NN  Implementation Train Data")
    plt.figure()
    plotit(Xtt,Ytt,clf=clf.predict)
    plt.title("K-NN  Implementation Test data")