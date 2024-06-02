# importing the required dependencies
import tensorflow as tf
import os
import cv2
import imghdr
import matplotlib.pyplot as plt
import numpy as np


data_dir='data'
x=os.listdir(os.path.join(data_dir,'happy'))

# gpus=tf.config.experimental.list_physical_devices('GPU')

# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu,True)


extensions=['jpeg','jpg','png','bmp']

# filtering out gibberish extensions files such as vectors or some other files the program might not be able to open

# for image_class in os.listdir(data_dir):
#     for image in os.listdir(os.path.join(data_dir,image_class)):
#         image_path=os.path.join(data_dir,image_class,image)
#         try:
#             img=cv2.imread(image_path)
#             tip=imghdr.what(image_path)
#             if tip not in extensions:
#                 print("image not in the extensions list()".format(image_path))
#                 os.remove(image_path)
#         except:
#             print("Issue with image()".format(image_path))
#             os.remove(image_path)


# loading the data and creating a data pipeline 

data=tf.keras.utils.image_dataset_from_directory('data')
data_iteator=data.as_numpy_iterator()
batch=data_iteator.next()


# scaling the data

scaled_data=data.map(lambda x,y:(x/255,y))


# fig,ax=plt.subplots(ncols=4,figsize=(20,20))
# for idx,img in enumerate(batch[0][:4]):
#     ax[idx].imshow(img.astype(int))
#     ax[idx].title.set_text(batch[1][idx])
#     plt.show()

# upon running the piece of code above, we find that 1 represents a sad person and 0 represents a happy person



# DATA PREPROCESSING
train_size=int(len(data)*.6)
val_size=int(len(data)*.2)+1
test_size=int(len(data)*.1)+1

train=data.take(train_size)
val=data.skip(train_size).take(val_size)
test=data.skip(train_size+val_size).take(test_size)

# BUILDING THE DEEP LEARNING MODEL

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model=Sequential()

model.add(Conv2D(16,(3,3),1,activation='relu',input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3),1,activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16,(3,3),1,activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam',loss=tf.losses.BinaryCrossentropy(),metrics=['accuracy'])



# # TRAINING THE DATA
logdir='logs'
tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist=model.fit(train,epochs=20,validation_data=val,callbacks=[tensorboard_callback])


# TESTING THE DATA

import cv2
# img=cv2.imread('sohappy.jpg')
# plt.imshow(img,cv2.COLOR_BGR2RGB)
# plt.show()

# resize=tf.image.resize(img,(256,256))
# plt.imshow(resize.numpy().astype(int))


# np.expand_dims(resize,0)
# pred=model.predict(np.expand_dims(resize/255,0))

# print(pred)

img=cv2.imread('ekaurhappy.jpg')
resize=tf.image.resize(img,(256,256))

np.expand_dims(resize,0)
pred=model.predict(np.expand_dims(resize/255,0))

print(pred)

