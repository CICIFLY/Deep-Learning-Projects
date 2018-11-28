# Imported Modules
import os
import random
import numpy as np
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
from progress.bar import ChargingBar as Bar

import warnings
warnings.filterwarnings("ignore",category=UserWarning)


class DataStruct:

	def __init__(self,name,path_src,path_dest,amt):

		self.name = name
		self.path_src = path_src
		self.path_dest = path_dest
		self.amt = amt


class DataBuild:

	def __init__(self,data_structs,transforms=None,myseed=None):


		self.data_structs = data_structs
		self.transforms = transforms

		self.available_transformations = {
			'rotate': self.random_rotation,
			'noise': self.random_noise,
			'horizontal_flip': self.horizontal_flip
		}

		random.seed(myseed)
		

	@staticmethod
	def random_rotation(image_array):
		random_degree = random.uniform(-25,25)
		return sk.transform.rotate(image_array,random_degree,mode='edge',preserve_range=True).astype(np.uint8)

	@staticmethod
	def random_noise(image_array):
		return sk.util.random_noise(image_array)

	@staticmethod
	def horizontal_flip(image_array):
		return np.flip(image_array,1)

	def run(self):

		for structobj in self.data_structs:
			imagepaths = [os.path.join(structobj.path_src,f) for f in os.listdir(structobj.path_src) if os.path.isfile(os.path.join(structobj.path_src,f))]

			self.transformData(imagepaths,structobj)


	def transformData(self,imagepaths,structobj):
		
		print("Generating Augmented Data for {}".format(structobj.name))
		bar = Bar('Processing',max=structobj.amt)
		for i in range(structobj.amt):

			# Grab a random image path from the image path list
			image_path = random.choice(imagepaths)
			image_to_transform = sk.io.imread(image_path,as_gray=True)
			transformed_image = image_to_transform

			if self.transforms is None:
				
				num_transformations_to_apply = random.randint(1,len(self.available_transformations))

				num_transformations = 0
				while num_transformations <= num_transformations_to_apply:
					#Apply our transformation
					key = random.choice(list(self.available_transformations))
					transformed_image = self.available_transformations[key](image_to_transform)
					num_transformations += 1

			else:
				for transform in transforms:
					#Apply each of the transforms on a image
					transformed_image = transform(image_to_transform)
					image_to_transform = transformed_image


			new_file_path = "{}/{}_{}.jpg".format(structobj.path_dest,structobj.name,i)
			io.imsave(new_file_path,transformed_image)

			bar.next(1)

		bar.finish()
		print("Done...")





# Our Main Method
if __name__ == '__main__':
             
	path = '/home/hscilab282-10/Desktop/Xi_Zhou/AI/11_6_2018_CNN/CNN_Face/'

    # here we can generate happy train, happy test, sad train,sad test
	happy = DataStruct(name="XZ_0",                                 
		                    path_src="{}happy_resize_gs".format(path),    # here raw_happy, raw_sad are resized and grayscale images
		                    path_dest="{}All_Faces".format(path),
		                    amt=30000)         # name must be different 

	sad = DataStruct(name="XZ_1",
		                    path_src="{}sad_resize_gs".format(path),
		                    path_dest="{}All_Faces".format(path),
		                    amt=30000)


	buildobj = DataBuild((happy, sad))
	buildobj.run()

	#buildobj1 = DataBuild((happytrain,),myseed=10101010)
	#buildobj2 = DataBuild((happytest,))
	#buildobj = DataBuild((happytrain, happytest), 
    #transforms=(DataBuild.random_noise,DataBuild.horizontal_flip,))
	
	