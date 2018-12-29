# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./figures/track_2_b_1.png "Visualization"
[image2]: ./figures/track_2_b_2.png "Visualization"
[image3]: ./writeup-img/output_18_1.png "Visualization"
[image4]: ./writeup-img/output_27_0.png "Visualization"
[image5]: ./writeup-img/output_34_0.png "Visualization"
[image6]: ./writeup-img/output_37_1.png "Visualization"
[image7]: ./writeup-img/output_42_0.png "Visualization"


Overview
---
I decided to go with NVIDA model as suggested with in the course. The model part is straight forward. The trick part is really generating good training and validation data. Recording the data in an intelligent way, using the supplied test data and curating and annotating the data is key to success. In addition to the test data set which was provided by udacity, I recorded for track 1 a full track in the reverse direction. For track 2 I recorded two rounds.


I decided not to use generator. On my system (iMac) the time to train the model increased by a factor of 20 compared to the version without a generator. As I have enough memory and not been able to use a GPU this was the most efficient approach for me.


Rubric Points
---
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

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














# -------------------------------------------
# REMOVE AFTER DONE
# -------------------------------------------


Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

