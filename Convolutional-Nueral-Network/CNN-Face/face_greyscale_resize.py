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


def gs_resize (name, num):
    home=os.getcwd()  # locate to the directory where the folder is 
    root1 = home+"/"+name+"_face/" # PATH HERE
    root2 = home+"/" + name +"_resize/" # PATH HERE
    root3 = home+"/" + name +"_resize_gs/" # PATH HERE


    data_list = [] 
    file_list = os.listdir(root1)  # get only files' name 

    print(len(file_list))
    for i in range(len(file_list)):
        #print(file_list)
        picture = root1 + file_list[i]  # get the file path + file name  
        resize = resize_file( picture , root2 + "XZ_"+str(num)+str(i)+".jpg", (28,28))
        grescale = greyscale_img(root2 + "XZ_"+str(num)+str(i)+".jpg", root3 + "XZ_"+str(num)+str(i)+".jpg")

name = ["happy", "sad"]
for i in range(len(name)):
    gs_resize(name[i],i)