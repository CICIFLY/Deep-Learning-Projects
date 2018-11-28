# for assignments
# read in all data and split
# shape of the hidden layers
# refrence materials 
# https://towardsdatascience.com/neural-network-on-iris-data-4e99601a42c8
# https://www.ibm.com/communities/analytics/watson-analytics-blog/watson-analytics-use-case-the-iris-data-set/
import numpy as np
import pandas as pd 
from sklearn import datasets

#iris = np.loadtxt("iris_data.dat")  # error when print out x.shape : ValueError: could not convert string to float: '5.1,3.5,1.4,0.2,setosa'
iris = datasets.load_iris()
X = iris.data   #(150,4)    
y_type = iris.target  #(150,)
#print(y_type.shape)
y=[]
for i in y_type:
    if i == 0:
        y.append((1,0,0)) # setota
    elif i == 1:
        y.append((0,1,0)) # versicolor
    else:
        y.append((0,0,1)) # virginica


b = np.array(y)
# print(b.shape )   # (150, 3)
# print(y_type.shape)  # (150, )

'''
np.random.seed(5)   # before the split line to obtain same result with every run.                                                       # do i need random.seed twice ????
# method 1 : use sklearn.model_selection.train_test_split twice. First to split to train, test and then split train again into validation and train. 
import sklearn 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
print("X_train.shape:",X_train.shape , "X_val.shape:",  X_val.shape , "X_test.shape:",  X_test.shape)    # (96, 4), (24, 4) ,(30, 4)

'''


# method 2: produces a 60%, 20%, 20% split for training, validation and test sets use pandas.
# path_to_file = "/home/hscilab282-10/Desktop/Xi_Zhou/AI/9_18_first_assignment/iris_data.dat"
# df = pd.read_csv(path_to_file , encoding = "utf-8")
# train, validate, test = np.split( df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])  #frac=1 means return all rows (in random order).
# print(train.shape, validate.shape, test.shape)  # (89, 5) (30, 5) (30, 5)



#ReLu function and its derivativ

def ReLu(x,deriv=False):   # leaky relu
    x_copy = np.copy(x)
    if(deriv==True):
        x_copy[x<0] = 0.1
        x_copy[x>=0] = 1
        return x_copy
    return np.maximum(0.1*x_copy,x_copy)   
   
   
#input data (instances),  got from Iris data ,5 for each type 
#output labels , each row stands for a type of flower 

#seed the pseudo-random no. generator
np.random.seed(1)

#synapses (weights)
syn0 = np.random.random((4,5))-1
syn1 = np.random.random((5,3))-1


#forward propagation, training
for j in range(600000):
    #layers
    l0 = X #inputs
    l1 = ReLu(np.dot(l0,syn0)) #hidden layer
    l2 = ReLu(np.dot(l1,syn1)) #output
    #back propagation
    l2_error = b - l2                                                                                                                       
    if(j % 10000) == 0:
        print ('Error '+str(np.mean(np.abs(l2_error))))
    l2_delta = l2_error*ReLu(l2,deriv=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error*ReLu(l1,deriv=True)
    
    syn1 += 0.0001 * l1.T.dot(l2_delta)
    syn0 += 0.0001 * l0.T.dot(l1_delta)
    
print ('Output after training:\n')
print ("l2: \n\n",l2)
print ("y :\n\n",y )
print("l2_error\n\n",l2_error)
