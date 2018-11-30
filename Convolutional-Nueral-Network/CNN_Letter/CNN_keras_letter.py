# reference 
# https://medium.com/@awjuliani/visualizing-neural-network-layer-activation-tensorflow-tutorial-d45f8bf7bbc4
# https://hackernoon.com/visualizing-parts-of-convolutional-neural-networks-using-keras-and-cats-5cc01b214e59
# https://github.com/erikreppel/visualizing_cnns/blob/master/visualize_cnns.ipynb

# using keras and convolutional neural network
# it invloves not only convert images into array but also add a label to the matrix

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # tells tensorflow to use CPU only

from tqdm import tqdm
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input,Dense,Conv2D,Flatten
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from skimage.io import imread,imshow
from PIL import Image
import numpy as np
# import tensorflow as tf
import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

# create the directories for later use 
vis_image_index = 0
l1 = '/home/hscilab282-10/Desktop/Xi_Zhou/AI/11_6_2018_CNN/CNN_Letter/l1'
l2 = '/home/hscilab282-10/Desktop/Xi_Zhou/AI/11_6_2018_CNN/CNN_Letter/l2'

# CHANGE TO SERVER PATH 
#l1 = '/home/hscilab282-10/Desktop/Xi_Zhou/AI/11_6_2018_CNN/CNN_Letter/l1'
#l2 = '/home/hscilab282-10/Desktop/Xi_Zhou/AI/11_6_2018_CNN/CNN_Letter/l2'

# load in the data 
dataset = np.loadtxt('Xi_letter_gen.txt', delimiter=' ') # mismatch in data, 00 contains all letters not A 
x = dataset[:,0:-1] # first 784 for x 
print("x" , x.shape)
y = dataset[:,-1] # last for y
print("y" , y.shape)
scaler = MinMaxScaler(feature_range=(0, 1)) 
x = scaler.fit_transform(x)

# split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
print ("X Train: ", x_train.shape)
print ("X Test: ", x_test.shape)

print ("Y Train: ", y_train.shape)
print ("Y Test: ", y_test.shape)

#X Train:  (28080, 784)
#X Test:  (3120, 784)
#Y Train:  (28080,)
#Y Test:  (3120,)


# normalize the data to [0,1] , otherwise it affects the accuracy when using different activation method
scaler = MinMaxScaler(feature_range=(0, 1))   
#x_test = scaler.fit_transform(x_test)
#x_train = scaler.fit_transform(x_train)
x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)


# reshape the data for mdoel    greyscale image 
print("First value of x_train shape " , x_train.shape[0])
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  #x _train[0] is the size of all x_train
x_test =  x_test.reshape(x_test.shape[0],28, 28, 1)     # reshape later, MinMaxScaler expected array<= 2.


# print(y_train[0])   output 5     # original result 

# one-hot encode target column y   one-hot encoding to make all numbers with long decimal to just bianary numbers (1,0...)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



# print(y_train[0])   output [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.] # after one-hot , the result , 
# for number 5 , 1 will be at index 5 

# build the model
model = Sequential()

model.add(Conv2D( 32, kernel_size=3, activation='relu',input_shape=(28,28,1),name='l1')) # layer 1  (28,28,1) input size , (26,26,32) output node size   kernel_size = 3 means a 3 by 3 mask 
intermediate_layer_model_1 = Model(inputs=model.input,
                                 outputs=model.get_layer('l1').output)


model.add(Conv2D( 16, kernel_size=3, activation='relu',name='l2')) # layer 2   (26,26,32) as input size , (24,24,16) is the output size


intermediate_layer_model_2 = Model(inputs=model.input,
                                 outputs=model.get_layer('l2').output)

intermediate_output_1 = intermediate_layer_model_1.predict(x_train[:4])
dim1 = intermediate_output_1.shape
print("output_layer1:\n\n",dim1) # (4,26,26,32)
print(dim1[-1])   # get 32
for i in tqdm(range(dim1[-1])):     # go to the last index
    fig = plt.figure()
    imshow(intermediate_output_1[vis_image_index,:,:,i],cmap='gray')
    plt.savefig('l1/figure_'+ str(i)+'.png')
    plt.close()

 
intermediate_output_2 = intermediate_layer_model_2.predict(x_train[:4])
dim2 = intermediate_output_2.shape
print("output_layer2:\n\n",dim2)
for i in tqdm(range(dim2[-1])):     # go to the last index
    fig = plt.figure()
    imshow(intermediate_output_2[vis_image_index,:,:,i],cmap='gray')
    plt.savefig('l2/figure_'+ str(i)+'.png')
    plt.close()


model.add(Flatten(name='l3'))  # (9216,1) is the output 

model.add(Dense(26,activation='softmax',name='l4'))   # here change 10 to 26 coz there are 26 classes(a-z)   (10,1) is the output

# compile the model
model.compile( optimizer = 'adam',    # 'sgd'  'adam'   'rmsprop'
    loss = 'categorical_crossentropy' , metrics = ['accuracy'])


# train the model
model.fit(x_train,y_train , validation_data = (x_test,y_test),epochs=1)
print('predictions:')
# print(model.predict(x_test[:4],end='\n\n'))
print(model.predict(x_test[:2]))
print('actual: ')
print(y_test[:2])





