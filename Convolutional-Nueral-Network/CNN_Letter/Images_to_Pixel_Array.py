# must run with python2  python2 Xi_Convert_to_Array.py
# this code will get the images in the directory to arrays and add extra label column to the matrix
import numpy as np
from PIL import Image
import os
import fnmatch

path = os.getcwd()
root = path +"/All_Letters_Data/"
data_list = [] 
file_list = os.listdir(root)  # get only files' name 


for i in range(len(file_list)):
    print(i)
    picture = root + file_list[i]  # get the file path + file name 
    print(picture)

    label = file_list[i][3:5]  # every letter has same length, 2 
    print (label)
    #print path_to_pic
    f = Image.open(picture)

    #print f.size
    pix_val = list(f.getdata())

    #print pix_val
    #print len(pix_val)

    pix_val.append(label)
    #print len(pix_val)

    #print pix_val
    data_list.append(pix_val)

    #break

data_list = np.array(data_list)

print (data_list.shape) # shape will be (31200, 785), 1 extra is for label

x = np.savetxt('Xi_letter_gen.txt', data_list, delimiter=" ", fmt = '%s')


