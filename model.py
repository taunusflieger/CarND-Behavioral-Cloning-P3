import os
import csv
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.models import Model, Sequential
from keras.layers import BatchNormalization, Input, Lambda, Activation, Flatten, Dense, Cropping2D, Convolution2D, Conv2D
from keras.regularizers import l2
from keras.optimizers import Adam


def resize(image):
    """
    Returns an image resized to match the input size of the network.
    :param image: Image represented as a numpy array.
    """
    return cv2.resize(image, (320, 160), interpolation=cv2.INTER_AREA)


def process_image(image):
    """
    Processing pipeline for the image. Currently only resizing
    """
    image = resize(image)
    return image


def load_data(name, file_path, X_data=None, y_data=None): 
    
    images_paths = []
    steering_angles = []

    path = os.path.dirname(file_path)
    print("Loading data: " + path)

    # Import data and remove most straight line driving using pandas:
    df = pd.read_csv(file_path,
                dtype={'center_img': np.str, 'left_img': np.str,
                        'right_img': np.str, 'steering': np.float64,
                        'throttle': np.float64, 'brake': np.float64,
                        'speed': np.float64}, header=0)

    
    df['steering'].plot.hist(title='Original steering distribution', bins=100)
    plt.savefig(fig_path + name + "_1.png")
    plt.gcf().clear()

    # Remove 92% of straight line driving data
    zero_indices = df[df['steering'] == 0].index
    remove_n = int(len(zero_indices)*0.92)
    df = df.drop(np.random.choice(zero_indices,size=remove_n,replace=False))

    # Remove 50% of steering action with a small angle
    f1= df['steering'] > -0.25 
    f2 = df['steering'] < 0.25
    zero_indices = df[f1 & f2].index
    remove_n = int(len(zero_indices)*0.5)
    df = df.drop(np.random.choice(zero_indices,size=remove_n,replace=False))

    df['steering'].plot.hist(title='Steering distribution after clean-up', bins=100)
    plt.savefig(fig_path + name + "_2.png")
    plt.gcf().clear()

    for index, row in df.iterrows():
        steering_center = float(row['steering'])   

        # create adjusted steering measurements for the side camera images
        correction = 0.25 # this is a parameter to tune (was .25)
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        img_center_path = path + '/IMG/' + row['center'].split('/')[-1]
        img_left_path = path + '/IMG/' + row['left'].split('/')[-1]
        img_right_path = path + '/IMG/' + row['right'].split('/')[-1]
            
        images_paths.extend([img_center_path, img_left_path, img_right_path])
        steering_angles.extend([steering_center, steering_left, steering_right])

    if X_data is not None and y_data is not None:
        X_data = np.append(X_data, images_paths)
        y_data = np.append(y_data, steering_angles)
    else:
        X_data = np.array(images_paths)
        y_data = np.array(steering_angles)    

    return X_data, y_data




def random_shadow(img, strength=0.50):
    """
    Random shawdow augmentation implementation as suggested by:
    https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
    """

    rows, cols, _ = img.shape

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    x_lo, x_hi = 0, rows

    rand_y = (cols * np.random.uniform(), cols * np.random.uniform())
    y_lo, y_hi = np.min(rand_y), np.max(rand_y)

    shadow_mask = 0 * img[:, :, 1]
    X_msk = np.mgrid[0:rows, 0:cols][0]
    Y_msk = np.mgrid[0:rows, 0:cols][1]
    shadow_mask[((X_msk - x_lo) * (y_lo - y_hi) - (x_hi - x_lo) * (Y_msk - y_hi) >= 0)] = 1

    if np.random.randint(2) == 1:
        random_bright = strength
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2) == 1:
            img[:, :, 1][cond1] = img[:, :, 1][cond1] * random_bright
        else:
            img[:, :, 1][cond0] = img[:, :, 1][cond0] * random_bright
    img = np.array(img).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HLS2RGB)

    return img


def disturb_brightness(img, strength=0.50):

    # will create a random brightness factor between [strenght, 1+strenght] to apply to the brightness channel
    rnd_brightness = strength + np.random.uniform()

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = np.array(img).astype(np.float64)
    img[:, :, 2] = np.clip(img[:, :, 2] * rnd_brightness, 0, 255)
    img = np.array(img).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    return img


def generate_data(X_samples, y_samples):
    images = []
    angles = []
    shuffle(X_samples, y_samples)
    for name, steering_angle in zip(X_samples, y_samples):
        augmented = False
        image = process_image(cv2.imread(name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        angles.append(steering_angle)          
      
       # Data augmentation
        if np.random.uniform(0., 1.) < 0.60:
            image = random_shadow(image)
            augmented = True
       
        if np.random.uniform(0., 1.) < 0.60:
            image = disturb_brightness(image)
            augmented = True
       
        if np.random.uniform(0., 1.) < 0.50:
            image = np.fliplr(image)
            steering_angle = -steering_angle
            augmented = True

        if augmented:
            images.append(image)
            angles.append(steering_angle)   
  
    X_train = np.array(images)
    y_train = np.array(angles)
    return  X_train, y_train



def get_model(input_shape):
    """
    This function will return a convolutional neural network as described on
    "End to End Learning for Self-Driving Cars" by NVIDIA
    http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """
    i_rows, i_cols, i_channels = input_shape

    inputs = Input(((i_rows, i_cols, i_channels)))

    x = Cropping2D(cropping=((30, 25), (0, 0)))(inputs)
    x = BatchNormalization()(x)

    x = Conv2D(24, kernel_size = (5, 5), padding='valid', subsample=(2, 2), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(36, kernel_size = (5, 5), padding='valid', subsample=(2, 2), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(48, kernel_size = (5, 5), padding='valid', subsample=(2, 2), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(64, kernel_size = (3, 3), padding='valid', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(64, kernel_size = (3, 3), activation='relu', padding='valid', use_bias=True)(x)
    x = Flatten()(x)
    x = Dense(100, kernel_regularizer=l2(reg), activation='relu', use_bias=True)(x)
    x = Dense(50, kernel_regularizer=l2(reg), activation='relu', use_bias=True)(x)
    x = Dense(10, kernel_regularizer=l2(reg), activation='relu', use_bias=True)(x)

    output = Dense(1, activation='relu', use_bias=True)(x)
    model = Model(input=inputs, output=output)

    return model



if __name__ == '__main__':
    fig_path = './figures/' 
    reg = 1e-3
    reg = 1e-3
    lr = 1e-3
    optimizer = Adam(lr=lr)  


    model = get_model(input_shape=(160, 320, 3))
    model.summary()
    
    # I trained the model with the default learning rate for Adam and it behaved
    # so well that I didn't feel I had much to gain by changing the initial Learning
    # rate.
    model.compile(optimizer=optimizer, loss='mse')


    # Load the data sets. We use the original udacity data set for track one
    # and a reverse recorded set for track one. For track two I've recorded
    # two rounds. For each track we have 2 rounds of recorded data.
    X_data, y_data = load_data('udacity_test_data', '/Volumes/Data 2/train-data/data/driving_log.csv')
    X_data, y_data = load_data('reverse_track_1', '/Volumes/Data 2/train-data/reverse-track/driving_log.csv', X_data, y_data)
    X_data, y_data = load_data('track_2_a', '/Volumes/Data 2/train-data/challange5/driving_log.csv', X_data, y_data)
    X_data, y_data = load_data('track_2_b', '/Volumes/Data 2/train-data/challange6/driving_log.csv', X_data, y_data)

    X_data, y_data = shuffle(X_data, y_data, random_state=14)

    # Generate the training and validation data (apply augmentation)
    X_train, y_train = generate_data(X_data, y_data)

    print("Size of dataset: " + str(len(X_train)))

    history_object = model.fit(X_train, y_train, validation_split=0.2, epochs=10, verbose=1)

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig(fig_path + "training.png")



    model.save('./Term1/CarND-Behavioral-Cloning-P3/model.h5')
