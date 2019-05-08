'''
The whole program is to build a nueral network model with both forward propogation and backward propogation.
The input is a randomly generated 1000 by 3 matrix.
The whole model includes 1 hidden layer with 4 nodes in the first hidden layer .
The output is 1000 by 2 matrix 
'''
import numpy as np
from numpy import random, exp, dot, array
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class ANN():
	def __init__(self):

		random.seed(1)
		self.weights_h0 = 2 * random.random((3,4)) -1
		self.weights_h1 = 2 * random.random((4,2)) -1
		#self.weights_ho = 2 * random.random((2,3)) -1


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
			l1 = self.summation_activation( l0, self.weights_h0)  # 4 by 3 * 3 by 1000(after transpose) = 4 by 1000 # to call the function, you need to use "self"
			l2 = self.summation_activation( l1, self.weights_h1)  # 3 by 4 * 4 by 1000 = 3 by 1000			

			# back propogation section  
			# error calculation.  error size and delta size should be the same 
			l2_error = output_data - l2
			mse = mean_squared_error(output_data , l2)

			# mean_squared_error is a default function in scikit learn library
			if n % 100 == 0: 
				error.append(mse)
			#plt.plot(mse)  can not plot here, should be outside of th loop
			l2_delta = l2_error  * self.__sigmoid_derivative(l2)  # go through aggregation and summation function


			l1_error = l2_delta.dot(self.weights_h1.T)  # self.weights_h0 won't be used 
			l1_delta =  l1_error * self.__sigmoid_derivative(l1)   # !!!!!!! here you can not use 'dot', must use multiply

			# 0.01 * l1.T.dot(l2_delta) and 0.01 * l0.T.dot(l1_delta) are 2 gradients for 2 matrics
			# 3 components : cost total, activation output, summation output. And error and activation should be multipied first 

			# adjust the weight	
			self.weights_h1 += 0.01 * l1.T.dot(l2_delta)
			self.weights_h0 += 0.01 * l0.T.dot(l1_delta)

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

	print("The initial weights for hidden layer 1 is:\n" ,neural_network.weights_h0 )    # after creating a class instance, do not need self here
	print("The initial weights for output layer is:\n" ,neural_network.weights_h1 ) 
	print('\n\n\n\n\n')  
 

	neural_network.train(input_data, output_data, 1000)

	print("The new weights is for hidden layer 1 :\n" ,neural_network.weights_h0 )    
	print("The initial weights for output layer is:\n" ,neural_network.weights_h1 )    



# notes : 
#no underscore or single underscore make the function public
# double underscore makes it private, which makes can not be used outside the scope like following
#ann1=ANN()
#ann1.__sigmoid()
#ann1.train()
