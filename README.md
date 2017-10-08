# **Behavioral Cloning** 



**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


###Preprocessing steps
1. 3 camera angles are recorded 
2. A correction factor of 0.2 is added in left and 0.2 subtracted right image
3. The track is flipped as a augmented image to get more balanced streeing when training the model

I used a simple Le-Net architecture implemented in Keras Architecture

Model Architecture and Training Strategy:
1. Input - (160,320,3) #Height, Width, RGB Layers
2. Normalise layer
3. Crop the layer - Top by 70, bottom by 25.
4. LeNet Conv1 with 5x5 filter, 6 output layers followed by pooling
5. LeNet Conv2 with 5X5 filter, 16 output layers followed by pooling
6. Flatten
7. Fully connected Dense layer - 120
8. Fully connected Dense layer - 84
Fully connected Dense layer - 1

The model used adam optimiser with intention to reduce "mean squared error". 

Epochs is reduced to 3 as post 3, it was seen to overfit the data.

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

The fully Autonoumous model "Abhi-selfdrivingcar.mp4"running complete track without missing the lane.

### Tried Attempts:
1. Dropouts from 0.1-0.5 in different layers. Contrary to improvement expectation, in the steep curves the car was straying sideways
2. Epochs - Above 5, always made car to stray sideways
3. Avoiding training of images with steering angle 0 and also (-0.25 to 0.25). The model behaved very good in normal curves but when the bridge comes(Texture change) it used to stray away from bridge

### Possible improvements:
Since data is the key for succesful training, these steps will be attempted and added in near future:
1. Brightness Augmentation
2. Shadow Augmentation
3. Different training track record and train
4. Additional image manipulation like perspective and affine transformation
5. Use of GoogLeNet or VGG transfer learning approaches
