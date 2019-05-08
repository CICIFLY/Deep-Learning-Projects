'''
The whole program is to build a nueral network model with both forward propogation and backward propogation.
The input is a randomly generated 1000 by 3 matrix.
The whole model includes 2 hidden layer with 4 nodes in the first hidden layer and 3 nodes in the second hidden layer .
The output is 1000 by 2 matrix 
'''
import numpy as np
from numpy import random, exp, dot, array
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class ANN():
    def __init__(self):

        # use random.seed() to make sure each time the number is generated the same 
        random.seed(1)

        # randomly created 3 weight matrices with values between -1 and 1 
        self.weights_h0 = 2 * random.random((3,4)) -1
        self.weights_h1 = 2 * random.random((4,3)) -1
        self.weights_h2 = 2 * random.random((3,2)) -1


    # define the sigmoid function for later use 
    def __sigmoid(self, x, deri = False ):

        '''
        This function takes input into the aggregation function which is sigmoid function. This would be used in the forward propogation
        '''
        return (1/(1 + exp(-x)))


    # define the sigmoid derivative function for later use , double underscore to make it private 
    def __sigmoid_derivative(self, x ):

        '''
        This function is used in backward propogation. Take 
        '''
        return ( x * (1-x) )    # have to use the sysmbol  " * "


    def summation_activation(self,input_data,weights):              
        '''
        This function would mutiple input_data and weights, then put the product into sigmoid function

        '''
        return self.__sigmoid(dot( input_data, weights))


    # train the neural network through trail and error
    # weights would be adjusted each time through the training process
    def train(self, input_data, output_data, n_iteration):
        '''
        This train function includes both forward propogation and backward propogation

        '''
        error = []
        for n in range(n_iteration):
            
            # forward propogation section
            # summation_activation includes the summation and aggreation functions
            # l1 is simiilar to h1
            l0 = input_data   # 1000 by 3 
            l1 = self.summation_activation( l0, self.weights_h0)  # 1000 by 3 * 3 by 4  = 1000 by 4 # to call the function, you need to use "self"
            l2 = self.summation_activation( l1, self.weights_h1)  # 1000 by 4 * 4 by 3 = 1000 by 3
            l3 = self.summation_activation( l2, self.weights_h2 ) # 1000 by 3 * 3 by 2 = 1000 by 2  ( same size with output now )
            

            # back propogation section  
            # error calculation
            # error size and delta size should be the same 
            l3_error = output_data - l3
            mse = mean_squared_error(output_data , l3)

            # mean_squared_error is a default function in scikit learn library
            if n % 100 == 0: 
                error.append(mse)
            #plt.plot(mse)  can not plot here, should be outside of th loop
            l3_delta = l3_error  * self.__sigmoid_derivative(l3)  # go through aggregation and summation function            
            
            l2_error = l3_delta.dot(self.weights_h2.T)
            l2_delta = l2_error * self.__sigmoid_derivative(l2)   # multipy not dot  !!!!!!!

            l1_error = l2_delta.dot(self.weights_h1.T)  # self.weights_h0 won't be used 
            l1_delta =  l1_error * self.__sigmoid_derivative(l1)   # !!!!!!! can not use 'dot',  multiply

            # adjustment is the gradient 
            # 3 components : cost total, activation output, summation output. And error and activation should be multipied first 

            # adjust the weight , 0.001 * l2.T.dot(l3_delta)  is the gradient 
            self.weights_h2 += 0.001 * l2.T.dot(l3_delta)   # 0.01 is learning rate.    w^(k+1) = w^k + delta w        learning rate * summation * 
            self.weights_h1 += 0.001 * l1.T.dot(l2_delta)
            self.weights_h0 += 0.001 * l0.T.dot(l1_delta)


        plt.plot(error)
        plt.title("The Distribution of Error during Training")
        plt.show()


if __name__ == "__main__":    

    
    input_data = array( 2 * random.random((1000,3)) -1 )
    output_data = array( 2 * random.random((1000,2)) -1)  # we need to hot encode it to (0,1), (1,0)
    for i in range(len(output_data)):   # rows
        for j in range(2):
            if output_data[i][j] < 0.5 :
                output_data[i][j] = 0
            else:
                output_data[i][j] = 1
    

    # initialize the neural network
    neural_network = ANN()

    print("The initial weights for hidden layer 1:\n" , neural_network.weights_h0 )    # after creating a class instance, do not need self here
    print("The initial weights for hidden layer 2:\n" , neural_network.weights_h1 )    
    print("The initial weights for output layer:\n" , neural_network.weights_h2 )   
    print('\n\n\n\n\n')  


    neural_network.train(input_data, output_data, 1000)

    print("The new weights for hidden layer 1:\n" ,neural_network.weights_h0 )    
    print("The new weights for hidden layer 2:\n" ,neural_network.weights_h1 )    
    print("The new weights for output layer:\n" ,neural_network.weights_h2 )    



# notes : 
#no underscore or single underscore make the function public
# double underscore makes it private, which makes can not be used outside the scope like following
#ann1=ANN()
#ann1.__sigmoid()
#ann1.train()
