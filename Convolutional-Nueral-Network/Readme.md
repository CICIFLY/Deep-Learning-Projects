# This folder contains 3 projects implemented in CNN to build image classfiers for smile face, sad face, 26 letters and 10 digits

# Workflow :
*	Augmented 31,200 image data based on 26 handwritten alphabets images, 
* converted the image data to matrix data after grayscale, 
* normalized the matrix,  
* split train and test set to get ready for the model
*	Built a convolutional neural network that can automatically classify 26 alphabetic letters

# Libraries Required
keras, scikit-learn, scikit-image,  numpy, tqdm, matplotlib, pillow 

# Datasets:
* MNIST image data
* Augmented face image data
* Augmented 26 letters image data 

# Lesson I learned:
The biggest lesson I learned from doing the projects is the original pictures must be clear enough for the model to learn.
Otherwise, trash in, trash out. The original accuracy was 0.038 before I redrew my images. 

