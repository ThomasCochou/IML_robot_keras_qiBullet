# coding=utf-8

#######################################################
# LOAD DATA
#######################################################

import os,glob
import numpy
from PIL import Image
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

# https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Initialisation d'une liste pour contenir les images
# et d'une liste pour contenir les personnalités (personalities_names)
raw_data =[]
personalities_names =[]
personalities_names_int=[]

reshape_raw_data = []

# Définir les chemins vers les images,
myRawData = 'chalamet_zemmour_300x300/*'

# Chargement des images
for one in glob.glob(myRawData):
   # Ajout de l'image
   data = Image.open(one)
   data = numpy.asarray(data)
   raw_data.append(data)
   # Ajout du personalities_names
   personalities_names.append(one.split("\\")[-1].split('_')[0])
   # Ajout de 0 pour Zemmour et de 1 pour Chalamet
   if(personalities_names[-1] == "zemmour") :
      personalities_names_int.append(0)
   else :
      personalities_names_int.append(1)


# Reformatage en numpy pour plus de facilité
raw_data = numpy.asarray(raw_data)
personalities_names_int = numpy.asarray(personalities_names_int)


# #######################################################
# # DATA GENERATION
# #######################################################

# https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/

generated_data = []
generated_personalities_names_int = []

# create data generator
datagen = ImageDataGenerator(width_shift_range=0.5, height_shift_range=0.5,horizontal_flip=True,rotation_range=90,brightness_range=[0.2,1.0])

x = 0

for data in raw_data :

   sample = numpy.expand_dims(data, 0)
   it = datagen.flow(sample, batch_size=1)

   # generate samples and plot
   for i in range(50):
      # generate batch of images
      batch = it.next()

      generated_data.append(batch[0])
      generated_personalities_names_int.append(personalities_names_int[x])
   
   x = x + 1

generated_data = numpy.asarray(generated_data)
generated_personalities_names_int = numpy.asarray(generated_personalities_names_int)


# #######################################################
# # DATA TRANSFORMATION
# #######################################################

# Separate data
X_train, X_test, y_train, y_test = train_test_split(generated_data, generated_personalities_names_int, test_size=0.3,random_state=109) # 70% training and 30% test

# Normalisation des images
X_train, X_test = X_train / 255.0, X_test / 255.0


#######################################################
# models
#######################################################

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, 3, activation='relu',input_shape=(300, 300,3)),
  tf.keras.layers.MaxPool2D((2,2)),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPool2D((2,2)),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPool2D((2,2)),                     
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history_train = model.fit(X_train, y_train, epochs=7)

model.evaluate(X_test,  y_test, verbose=2)

model.save('model_keras')



#######################################################
# plot history
#######################################################

plt.plot(history_train.history['accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history_train.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#######################################################
# predict
#######################################################


# Initialisation d'une liste pour contenir les images à prédire
data_to_guess =[]

# Définir les chemins vers les images à prédire,
myDataToGuess = 'chalamet_zemmour_new/chalamet_zemmour_new_300x300/*'

# Chargement des images à prédire
for one in glob.glob(myDataToGuess):
   # Ajout de l'image
   data = Image.open(one)
   data = numpy.asarray(data)
   data_to_guess.append(data)
# Reformatage en numpy pour plus de facilité
data_to_guess = numpy.asarray(data_to_guess)

# Normalisation des images
data_to_guess = data_to_guess / 255.0

y_pred=model.predict(data_to_guess)


# #######################################################
# # display prediction
# #######################################################

# Un petit graphique pour illustrer
fig = plt.figure(figsize=(7,7))

for i in range(6):

   plt.subplot(330+i+1)

   plt.imshow(numpy.fliplr(data_to_guess[i]))

   if(y_pred[i][0] >= 0.5):
      plt.title("zemmour p=" + str(y_pred[i][0]))
   else:
      plt.title("chalamet p=" + str(y_pred[i][1]))

   plt.axis('off')
# Visualiser le plot
plt.show()
