import numpy as np
from PIL import Image
import os
import fnmatch


def resize_file(in_file, out_file, size):  # should be save into different dir every time and fetch from new dir 
    img=Image.open(in_file)
    s_img=img.resize(size,Image.ANTIALIAS)
    s_img.save(out_file)
    s_img.close()


# grayscale all the images 
def greyscale_img(in_file, out_file):
    img_gs_small = Image.open(in_file).convert('L')  
    img_gs_small.save(out_file)   # how to save all the photoes into a specific directory ????
    img_gs_small.close()


def gs_resize ():
    home=os.getcwd()  # locate to the directory where the folder is 
    root1 = home+"/Letters/" # PATH HERE
    root2 = home+"/Letters_resize/" # PATH HERE
    root3 = home+"/Letters_resize_gs/" # PATH HERE

    data_list = [] 
    file_list = os.listdir(root1)  # get only files' name 

    print(len(file_list))
    for i in range(len(file_list)):
        #print(file_list)
        # picture = root1 + file_list[i]  # get the file path + file name    unlike face data, the name does not need to be changed 
        resize = resize_file( root1 + file_list[i]  , root2 + file_list[i], (28,28))
        grescale = greyscale_img(root2 + file_list[i], root3 + file_list[i])


gs_resize()