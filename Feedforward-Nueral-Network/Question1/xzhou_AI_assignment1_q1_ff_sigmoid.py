import numpy as np

# linear activation function : identity fucntion  
def sigmoid_function(x,deriv = False) :
    if(deriv==True):
        return (x*(1-x))
    return (1/(1+ np.exp(-x)))
   
   
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

#synapses (weights)
syn0 = np.random.random((2,3))-1
syn1 = np.random.random((3,1))-1


#forward propagation, training
for j in range(60000):
    #layers
    l0 = x #inputs
    l1 = sigmoid_function(np.dot(l0,syn0)) #hidden layer
    l2 = sigmoid_function(np.dot(l1,syn1)) #output
    #back propagation
    l2_error = y_normalized - l2
    if(j % 10000) == 0:
        print ('Error '+str(np.mean(np.abs(l2_error))))
    l2_delta = l2_error * sigmoid_function(l2,deriv=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * sigmoid_function(l1,deriv=True)
    
    syn1 += 0.001 * l1.T.dot(l2_delta)
    syn0 += 0.001 * l0.T.dot(l1_delta)
    
print ('Output after training:\n')
print ("l2: \n",l2 )
print ("y :\n ",y_normalized )          
print ("l2 error :\n",l2_error)


    
