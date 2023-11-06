#importing the dependencies
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

#PREPROCESSING THE TRAININNG SET AND TEST SET

#training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training',
                                                 target_size = (124, 124),
                                                 batch_size = 300,
                                                 class_mode = 'binary')

#test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/Validation',
                                            target_size = (124, 124),
                                            batch_size = 300,
                                            class_mode = 'binary')

#BUILDING THE CNN MODEL
cnn = tf.keras.models.Sequential()

#convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[124,124,3]))

#polling layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#flatterning layer
cnn.add(tf.keras.layers.Flatten())

#full connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

#output layers
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

#COMPAILING THE TRAINING MODEL

#comapiling
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#training the model
cnn.fit(x=training_set,validation_data=test_set,epochs=25)



from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/female1.jpg', target_size = (124, 124))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'male'

else:
  prediction = 'female'

print(prediction)

