import csv
import cv2
import numpy as np

lines=[]
with open('./data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    #next(reader) 
    for line in reader:
        lines.append(line)
    del(lines[0])	
images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = line[3]
    measurements.append(measurement)
	
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, AveragePooling2D, Cropping2D, Convolution2D
from keras.layers.pooling import MaxPooling2D

def preprocess(image):
    import tensorflow as tf
    resized = tf.image.resize_images(image, (80, 160))
    normalized = resized/255.0 - 0.5
    return normalized
def resize_function(input):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(input, (80,160))


model = Sequential()
#model.add(AveragePooling2D(pool_size=(2, 2), strides=(2,2), border_mode='valid', dim_ordering='default'))
model.add(Lambda(lambda x: (x / 255.0)-0.5 , input_shape=(160,320,3)))
#model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
#model.add(Lambda(resize_function, input_shape=(160, 320, 3), output_shape=(80, 160,3)))
#model = Sequential()
#model.add(Lambda(lambda x: preprocess(x), input_shape=(3, 160, 320), output_shape=(3, 80, 160)))
#model.add(Cropping2D(cropping=cropping_shape))
#model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))

model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(16,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True,nb_epoch=5)
model.save('model.h5')
