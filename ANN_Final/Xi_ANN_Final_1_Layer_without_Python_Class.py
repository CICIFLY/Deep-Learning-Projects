'''
The whole program is to build a nueral network model with both forward propogation and backward propogation.
The input is a randomly generated 1000 by 3 matrix.
The whole model includes 1 hidden layer with 4 nodes in the first hidden layer .
The output is 1000 by 2 matrix 
'''

import numpy as np
import matplotlib.pyplot as plt 

#sigmoid function and its derivative
def nonlin(x,deriv=False):
   if(deriv==True):
       return(x*(1-x))
   return(1/(1+np.exp(-x)))
   
#input data (instances)
x = np.random.random((1000,3))

#output labels
y = np.random.random((1000,2))

#seed the pseudo-random no. generator
np.random.seed(1)

print("input size: " , x.shape)
print("output size: " , y.shape)


#synapses (weights)
syn0 = np.random.random((3,4))-1
syn1 = np.random.random((4,2))-1
print ("Hidden Layer 1 weight matrix:\n\n",syn0.shape)
print ("Hidden Layer 2 weight matrix:\n\n",syn1.shape)

#forward propagation, training
error = []
for j in range(60000):
    #layers
    l0 = x #inputs
    l1 = nonlin(np.dot(l0,syn0)) #hidden layer
    l2 = nonlin(np.dot(l1,syn1)) #output
    
    #back propagation
    l2_error = y - l2
    if j%1000 ==0 :
        mse = round((np.mean(np.abs(l2_error))),2)
        error.append(mse)
    l2_delta = l2_error * nonlin(l2,deriv=True)  

    l1_error = l2_delta.dot(syn1.T)
    print("l1 shape " , l1.shape)
    print("l1_error shape ", l1_error.shape)
    l1_delta = l1_error * nonlin(l1,deriv=True)


    syn1 += 0.001 * l1.T.dot(l2_delta)
    syn0 += 0.001 * l0.T.dot(l1_delta)
    
print ('Output after training')
print(l2_error.shape)
print(l2_delta.shape)
print(l1_error.shape)
print(l1_delta.shape)

plt.plot(error)
plt.title("The Distribution of Error during Training")
plt.show()
#print (l2)
#print (y)
#print (l1)
#print (syn0)
#print (syn1)
#print (l2_error)
    
