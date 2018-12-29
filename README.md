# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./figures/track_2_b_1.png "Visualization"
[image2]: ./figures/track_2_b_2.png "Visualization"
[image3]: ./figures/NVIDIA.png "NVIDIA Model"
[image4]: ./figures/center1.jpg  "Visualization"
[image5]: ./figures/center2.jpg  "Visualization"
[image6]: ./figures/reverse.jpg  "Visualization"
[image7]: ./figures/shadow.png  "Visualization"
[image8]: ./figures/brightness.png  "Visualization"
[image9]: ./figures/flipped.png  "Visualization"

Overview
---
I decided to go with NVIDA model as suggested with in the course. The model part is straight forward. The trick part is really generating good training and validation data. Recording the data in an intelligent way, using the supplied test data and curating and annotating the data is key to success. In addition to the test data set which was provided by udacity, I recorded for track 1 a full track in the reverse direction. For track 2 I recorded two rounds.


I decided not to use generator. On my system (iMac) the time to train the model increased by a factor of 20 compared to the version without a generator. As I have enough memory and not been able to use a GPU this was the most efficient approach for me.


Rubric Points
---
Here I will consider the rubric points individually and describe how I addressed each point in my implementation

#### Files required:
Videos: [Video Track1](https://github.com/taunusflieger/CarND-Behavioral-Cloning-P3/blob/master/track1.mp4) [Video Track2](https://github.com/taunusflieger/CarND-Behavioral-Cloning-P3/blob/master/track2.mp4) 

## Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 177-215) 


The model I’ve implemented is the following

```sh
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 105, 320, 3)       0
_________________________________________________________________
batch_normalization_1 (Batch (None, 105, 320, 3)       12
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 51, 158, 24)       1800
_________________________________________________________________
batch_normalization_2 (Batch (None, 51, 158, 24)       96
_________________________________________________________________
activation_1 (Activation)    (None, 51, 158, 24)       0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 24, 77, 36)        21600
_________________________________________________________________
batch_normalization_3 (Batch (None, 24, 77, 36)        144
_________________________________________________________________
activation_2 (Activation)    (None, 24, 77, 36)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 10, 37, 48)        43200
_________________________________________________________________
batch_normalization_4 (Batch (None, 10, 37, 48)        192
_________________________________________________________________
activation_3 (Activation)    (None, 10, 37, 48)        0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 8, 35, 64)         27648
_________________________________________________________________
batch_normalization_5 (Batch (None, 8, 35, 64)         256
_________________________________________________________________
activation_4 (Activation)    (None, 8, 35, 64)         0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 6, 33, 64)         36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 12672)             0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               1267300
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 1,404,747
Trainable params: 1,404,397
Non-trainable params: 350
...
```


#### 2. Attempts to reduce overfitting in the model

During my experiments it turned out that images with a zero steering angle create a lot of noise and doesn’t help the model to learn very well how to deal with curves. I decided to remove 92% of the data which doesn’t contain any curve steering information. I also decided to remove 50% of the data which contains only very limited steering (i.e. below .25 degree)
As an example the below figure shows the distribution of the steering angle before and after applying the above approach. Both figures are based on the recordings for track 2.

![alt text][image1]

![alt text][image2]

I’ve annotated the data by adding a certain percentage of additional data generated from the original data:
1.	Random shadows (60%). Here the images are annotated with a randomly generated shadow overlay
2.	Random Brightness (60%). Here brightness distribution is randomly changed
3.	Flipped images (50%). The images are flipped (also the steering angle is multiplied by -1.0)	

This approach increased the amount of training data to 64k training data points and 16k validation datapoints
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track for both track 1 and the challenging track 2. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 291).

#### 4. Appropriate training data

In addition to the test data set which was provided by udacity, I recorded for track 1 a full track in the reverse direction. For track 2 I recorded two rounds. For all the recordings I tried to stay in the middle of the track with a smooth steering.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I decided to go with NVIDA model as suggested with in the course. The model part is straight forward. The trick part is really generating good training and validation data. Recording the data in an intelligent way, using the supplied test data and curating and annotating the data is key to success. In addition to the test data set which was provided by udacity, I recorded for track 1 a full track in the reverse direction. For track 2 I recorded two rounds.

I decided not to use generator. On my system (iMac) the time to train the model increased by a factor of 20 compared to the version without a generator. As I have enough memory and not been able to use a GPU this was the most efficient approach for me.

To combat the overfitting, I modified the model and added several BatchNormalization layers

At the end of the process, the vehicle is able to drive autonomously around both tracks without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 177-215) consisted of a convolution neural network which follows the NVIDIA approach

![alt text][image3]


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image4]

![alt text][image5]

I then recorded the vehicle driving track 1 in the reverse direction:

![alt text][image6]


To augment the data sat, I also flipped images, added random shadow and increased randomly brightness. This all allowed me to generate more training and validation data out of the existing recorded data. 

Shadow example:
![alt text][image7]

Brigthness example:
![alt text][image8]

Flipped example
![alt text][image9]

