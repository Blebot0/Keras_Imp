import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

#Model

img_width, img_height = 28, 28
batch_size = 250
epochs = 25
classes = 10 ## {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
validation_split = 0.2
verbosity = 1

(input_train, target_train), (input_test, target_test) = mnist.load_data()

#Reshaping as channels first because of Keras backend or others if there is any other backend

if K.image_data_format() == 'channels_first':
    input_train = input_train.reshape(input_train[0], 1,  img_width, img_height)
    input_test = input_test.reshape(input_test[0], 1, img_width, img_height)
    input_shape = (1, img_width, img_height)
else:
    input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 1)
    input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 1)
    input_shape = (img_width, img_height, 1)

#Parsing data as float32
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

#Convert BGR to grayscale

input_test = input_test/255
input_train = input_train/255

target_train = keras.utils.to_categorical(target_train, classes)
target_test = keras.utils.to_categorical(target_test, classes)

#Model

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(classes, activation='softmax'))

#Compile
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

#Fit data to model

model.fit(input_train, target_train, batch_size=batch_size, epochs=epochs, verbose=verbosity, validation_split =validation_split)

#generate score

score = model.evaluate(input_test, target_test, verbose=0)
print('Test loss: {0} / Test accuracy: {1}'.format(score[0], score[1]))


