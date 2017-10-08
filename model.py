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
    for i in range(3):
        #print(line[i])
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'data/data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
    correction = 0.2
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement + correction)
    measurements.append(measurement - correction)


augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image =  cv2.flip(image, 1)
    flipped_measurement = float(measurement) * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)


#print(type(augmented_measurements))
#print(type(augmented_images))
X_train = np.array(augmented_images )
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, AveragePooling2D, Cropping2D, Convolution2D, Dropout
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
model.add(Lambda(lambda x: (x / 255.0)-0.5 , input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(16,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True,nb_epoch=3)
model.save('model.h5')
