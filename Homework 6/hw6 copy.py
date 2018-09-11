import tensorflow as tf
import numpy as np
import keras as keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.cross_validation import StratifiedShuffleSplit
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.regularizers import l1, Regularizer

# ### Task 3: Convolutional Neural Net on SVHN dataset
# First to read in the data:

import scipy.io as sio
from scipy import sparse

train_mat = sio.loadmat("train_32x32.mat")
test_mat = sio.loadmat("test_32x32.mat")

X_train = train_mat['X']
y_train = train_mat['y']
X_test = test_mat['X']
y_test = test_mat['y']
X_train.shape

np.unique(y_train)
# After checking out the size and orientation of the dataset, we transpose/reshape the elements to make sure the final format maintains original picture composition

batch_size = 128
num_classes = 10
epochs = 12

X_train, y_train = X_train.transpose((3,0,1,2)), y_train[:,0]
X_test, y_test = X_test.transpose((3,0,1,2)), y_test[:,0]

print("Training Set", X_train.shape)

# Next making sure the labels are indexed properly and one-hot encoded:

y_train[y_train == 10] = 0
y_test[y_test == 10] = 0

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# I should note that I did plot the pictures to make sure they were reshaped properly 
# (as explained in the homework instructions) but due to issues with the "Backend" module of matplotlib, 
# I couldn't get the backend "TKagg" to work (which would allow me to show plots in a window in the notebook).
# So, I instead switched the backend to "agg" which allows me to save the figure. 
# Thus, to summarize, there is no plot of the pictures in this notebook because 
# I couldn't figure out how to make "TKagg" work on the Google Cloud VM, 
# but the plot of the images was saved as "initialimage.png" and I have submitted 
# it along with the notebooks as proof that I did re-configure the data as required.

# Now to create and train the base model. 

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)
num_classes = 10

cnn = Sequential()
cnn.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(.5))
cnn.add(Conv2D(32, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Flatten())
cnn.add(Dropout(.5))
cnn.add(Dense(64, activation='relu'))
cnn.add(Dropout(.5))
cnn.add(Dense(num_classes, activation='softmax'))

cnn.summary()

cnn.compile("adam", "categorical_crossentropy", metrics=['accuracy'])

history_cnn = cnn.fit(X_train, y_train,
                      batch_size=128, epochs=20, verbose=1, validation_split=.1)

# I'm really confused as to why the accuracy is so low, considering I did reshape the images as needed 
# (as you can see in "initialimage"). I tried a couple variations of the base model, initially without 
# dropout, with different number of hidden layers, and I couldn't get the accuracy to rise above 0.1893 
# (and I don't know why).(Note: I didn't show these models because I interrupted them partway through when 
# I saw no progress was being made). Since I'm pretty certain the issue isn't with the image preprocessing, 
# I have to assume the error is with the model, but I don't think anything I did is out of the ordinary. 
# Below see the results of the test evaluation.

cnn.evaluate(X_test, y_test)

from keras.layers import BatchNormalization

num_classes = 10
cnn_small_bn = Sequential()
cnn_small_bn.add(Conv2D(8, kernel_size=(3, 3),
                 input_shape=input_shape))
cnn_small_bn.add(Activation("relu"))
cnn_small_bn.add(BatchNormalization())
cnn_small_bn.add(MaxPooling2D(pool_size=(2, 2)))
cnn_small_bn.add(Conv2D(8, (3, 3)))
cnn_small_bn.add(Activation("relu"))
cnn_small_bn.add(BatchNormalization())
cnn_small_bn.add(MaxPooling2D(pool_size=(2, 2)))
cnn_small_bn.add(Flatten())
cnn_small_bn.add(Dense(64, activation='relu'))
cnn_small_bn.add(Dense(num_classes, activation='softmax'))

cnn_small_bn.compile("adam", "categorical_crossentropy", metrics=['accuracy'])

history_cnn_small_bn = cnn_small_bn.fit(X_train, y_train,
                                        batch_size=128, epochs=10, verbose=1, validation_split=.1)

cnn_small_bn.evaluate(X_test, y_test)

# As you can see in the results above, the training accuracy is much higher now 
# (which makes sense since batch normalization is supposed to increase accuracy), 
# but this further supports my hypothesis that the error with the previous model's accuracy 
# results was not a result of pre-processing, but rather the model. (Although, I don't know 
# what the model's error was, since this batch normalization model is very similar to the previous 
# base model). But let the test results below show that I did train a convolution neural net that 
# met the desired test accuracy.



### Task 4: Pre-trained Convolutional Neural Network
from keras import applications 
from keras.preprocessing import image as im
# load the VGG16 network
model = applications.VGG16(include_top=False, weights='imagenet')
model.summary()

import os
from keras.applications.vgg16 import preprocess_input

for filename in os.listdir("images"):
    if filename.endswith(".jpg"):
        img = im.load_img("./images/"+filename, target_size=(224,224))
        data = im.img_to_array(img)
        data = preprocess_input(data)
        np.save(filename, data)

# As you can see above, I ran out of disk space on my Google Cloud GPU and although I was able to resize
# the disk, I was not able to resize the file system so that it adjusts for the increased disk space 
# (and I did try to google the answer but all the tricks I attempted were not helpful). Since I seemed to 
# have hit a dead end with Google Cloud GPU, I thought about running this on my local disk but I don't want 
# to try it because many people have reported issues with their notebooks crashing while trying to run it 
# locally. Thus, I have decided to hit a middle ground of showing you the code I would have used next to 
# finish pre-processing and running the model on the data, so that you might get an impression of how I 
# would have approached this task:

# Once I had properly loaded and saved the image data, I would have split it into test and train, as shown 
# below, then generated the classification labels and fit the logistic regression model on the train set 
# and then used it to predict the test set.

X_train, X_test = train_test_split(X_pre)
features = model.predict(X_train)
print(X.shape)
features_ = features.reshape(200, -1)

from sklearn.linear_model import LogisticRegressionCV

lr = LogisticRegressionCV().fit(X_train, y_train)
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))





























