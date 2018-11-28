
# perceptron means single node
import numpy as np
'''
def identity(x,deriv = False) : # identity fucntion       
        return 1
    return x 
'''
#ReLu function and its derivativ
def ReLu(x,deriv=False):   # relu
    x_copy = np.copy(x)
    if(deriv==True):
        x_copy[x<0]=0
        x_copy[x>=0]=1
        return x_copy
    return np.maximum(0,x_copy)



#input data (instances)
x = np.array([[0.374186, 0.904727],          
             [ 0.397672,  0.074281],        
             [0.387948,  0.153990],          
             [0.541511,  0.148609],          
             [0.366906,  0.760656],      
             [0.389160,  0.570398],          
             [0.892068,  0.926974],         
             [0.507588,  0.752778],         
             [0.316727,  0.477287],         
             [0.727478,  0.414801] ])       

#output labels
y = np.array([[166], [221], [244], [61], [190], [131], [164], [216], [134], [146]])
y_normalized = y/244



#seed the pseudo-random no. generator
np.random.seed(1)

#print ("x:\n" , x)
#print ("y:\n" ,y)

#synapses (weights)
syn0 = np.random.random((2,1))-1  
#print ("syn0:\n" ,syn0)

#forward propagation, training
for j in range(60000):
    # single layer 
    l0 = x #inputs
    l1 = ReLu(np.dot(l0,syn0)) #hidden layer
    
    #back propagation
    l1_error = y_normalized  - l1
    if(j % 10000) == 0:
       print('Error '+ str(np.mean(np.abs(l1_error))))
    l1_delta = l1_error*ReLu(l1,deriv=True)   # without learning rate , go too fast, will diverge 
    syn0 +=  0.0001 *l0.T.dot(l1_delta)   # learning rate : 0.001 , always comes with dot product 
    
print ('Output after training')
print ("l1:\n" , l1)

print ("y_normalized :\n",y_normalized )
print ("syn0:\n",syn0) # three parameters 

print ("l1_error:\n",l1_error)
