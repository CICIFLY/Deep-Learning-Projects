import numpy as np
pima = np.loadtxt("pima_clean.txt")

# the following code is to set up x and y 

for i in range(8):
    pima[:,i] = pima[:,i] / np.max(pima[:,i])  # normalize x    

X = pima[:,0:8]
#print(X)

b_trans = pima[:,8]
y = []              
for i in b_trans:                                   # transfrom y 
    if i == 0:
        y.append((0,1)) # without diabetes
    else:
        y.append((1,0)) # with diabetes
  
b = np.array(y)  

#Sigmoid function and its derivative
def sigm(x,deriv=False):
  if(deriv==True):
    return(x*(1-x))
  return(1/(1+np.exp(-x)))
   
   
   
#input data (instances),  got from Iris data ,5 for each type 
#output labels , each row stands for a type of flower 

#seed the pseudo-random no. generator
np.random.seed(1)

print("For Sigmoid Function:")
for i in range(6,11):
    print("The hidden layer is:",i)
    syn0 = np.random.random((8,i))-1
    syn1 = np.random.random((i,2))-1


    #forward propagation, training
    for j in range(60000):
    #layers
        l0 = X #inputs
        l1 = sigm(np.dot(l0,syn0)) #hidden layer
        l2 = sigm(np.dot(l1,syn1)) #output
        #back propagation
        l2_error = b - l2                                                                                                                       
        if(j % 10000) == 0:
            print ('Error '+str(np.mean(np.abs(l2_error))))
        l2_delta = l2_error * sigm(l2,deriv=True)
        l1_error = l2_delta.dot(syn1.T)
        l1_delta = l1_error * sigm(l1,deriv=True)

        syn1 += 0.01 * l1.T.dot(l2_delta)
        syn0 += 0.01 * l0.T.dot(l1_delta)
    
    print ('Output after training:\n')
    print ("l2: \n\n",l2)
    print ("y :\n\n",y )
    print("l2_error\n\n",l2_error,"\n\n\n\n")
