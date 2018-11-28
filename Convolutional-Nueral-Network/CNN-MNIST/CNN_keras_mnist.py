# reference 
# https://medium.com/@awjuliani/visualizing-neural-network-layer-activation-tensorflow-tutorial-d45f8bf7bbc4
# https://hackernoon.com/visualizing-parts-of-convolutional-neural-networks-using-keras-and-cats-5cc01b214e59
# https://github.com/erikreppel/visualizing_cnns/blob/master/visualize_cnns.ipynb

# using keras and convolutional neural network

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # tells tensorflow to use CPU only

from tqdm import tqdm
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input,Dense , Conv2D,Flatten
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from skimage.io import imread,imshow

vis_image_index = 0
l1 = "/home/hscilab282-10/Desktop/Xi_Zhou/AI/10_30_2018_CNN/l1"
l2 = "/home/hscilab282-10/Desktop/Xi_Zhou/AI/10_30_2018_CNN/l2"



# load the data into test and train set
(x_train,y_train),(x_test,y_test) = mnist.load_data()
# print(x_train.shape,y_train.shape,x_test.shape,y_test.shape) 
# (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)



# plt.imshow(x_train[0],cmap='gray')
# plt.show()

# reshape the data for mdoel    greyscale image 
x_train = x_train.reshape(60000, 28, 28, 1)
x_test =  x_test.reshape(10000,28, 28, 1)

# print(y_train[0])   output 5     # original result 

# one-hot encode target column y   one-hot encoding to make all numbers with long decimal to just bianary numbers (1,0...)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train[0])   output [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.] # after one-hot , the result , 
# for number 5 , 1 will be at index 5 

# build the model
model = Sequential()

model.add(Conv2D( 32, kernel_size=3, activation='sigmoid',input_shape=(28,28,1),name='l1')) # layer 1  (28,28,1) input size , (26,26,32) output node size   kernel_size = 3 means a 3 by 3 mask 
intermediate_layer_model_1 = Model(inputs=model.input,
                                 outputs=model.get_layer('l1').output)

model.add(Conv2D( 16, kernel_size=3, activation='sigmoid',name='l2')) # layer 2   (26,26,32) as input size , (24,24,16) is the output size
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
model.add(Dense(10,activation='softmax',name='l4'))   # number of output nodes 64 to 32 to 10    (10,1) is the output

# compile the model
model.compile( optimizer = 'adam',    # 'sgd'
	loss = 'categorical_crossentropy' , metrics = ['accuracy'])


# train the model
model.fit(x_train,y_train , validation_data = (x_test,y_test),epochs=1)
print('predictions:')
# print(model.predict(x_test[:4],end='\n\n'))
print(model.predict(x_test[:4]))
print('actual: ')
print(y_test[:4])

# sgd loss: 0.4979 - acc: 0.8509 - val_loss: 0.2291 - val_acc: 0.9343
# adam loss:  loss: 0.2834 - acc: 0.9158 - val_loss: 0.0789 - val_acc: 0.9762

