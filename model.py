import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D, Cropping2D, Lambda, Convolution2D,Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam

flags = tf.app.flags
FLAGS = flags.FLAGS

#Usage
#python model.py --IMG_dir data/ --CSV_file data/driving_log.csv --epochs 10 --batch_size 64 --lrate 0.001

# command line flags
flags.DEFINE_string('IMG_dir', 'data/', "The directory of images")
flags.DEFINE_string('CSV_file', 'data/driving_log.csv', "The csv file for training data")
flags.DEFINE_integer('epochs', 10, "The number of epochs")
flags.DEFINE_integer('batch_size', 64, "The batch size")
flags.DEFINE_integer('lrate', 0.0001, "The learning rate")

def read_imgs(img_paths):
    
    #read images
    imgs = np.empty([len(img_paths), 160, 320, 3])
    for i, path in enumerate(img_paths):
        imgs[i] = imread(FLAGS.IMG_dir+path)
    return imgs


def preprocess(b_imgs):
    
    resized_imgs = np.empty([len(b_imgs), 40,160,3])
    for i, img in enumerate(b_imgs):
        image_array = np.asarray(img)
        image_array = image_array[59:138:2, 0:-1:2, :]
        transformed_image_array = image_array.reshape((1, 40, 160, 3))
        # Normalize pixels to values from -1.0 to 1.0
        transformed_image_array = (transformed_image_array / 127.5) - 1.0
        resized_imgs[i] = transformed_image_array
    return resized_imgs

def flip(images, angles):
    
    #flip images and angles 
    flipped_images = np.empty_like(images)
    flipped_angles = np.empty_like(angles)
    for i, (img, angle) in enumerate(zip(images, angles)):
        if np.random.choice(2):
            flipped_images[i] = np.fliplr(img)
            flipped_angles[i] = angle * -1
        else:
            flipped_images[i] = img
            flipped_angles[i] = angle

    return flipped_images, flipped_angles
    
def gen_batch(imgs, angles, batch_size):
    
    #imgs: The input images.
    #angles: The steering angles associated with each image.
    #batch_size: The size of each minibatch.
    #yield: A tuple (images, angles), where both images and angles have batch_size elements.
    
    num_of_elements = len(imgs)

    while True:
        indeces = np.random.choice(num_of_elements, batch_size)
        batch_imgs, batch_angles = read_imgs(imgs[indeces]), angles[indeces].astype(float)
        batch_imgs = preprocess(batch_imgs)
        batch_imgs, batch_angles = flip(batch_imgs, batch_angles)
        #print('in batch: ', batch_imgs[0].shape)
        yield batch_imgs, batch_angles   
       
def main(_):
    
    # ***load data
    with open(FLAGS.CSV_file, 'r') as csvf:
        try:
            reader = csv.reader(csvf)
            data = np.array([row for row in reader]) 
        finally:
            csvf.close()
    
    Xt_center, Xt_left, Xt_right, steering, throttle, brake, speed = list(zip(*data))
    
    Xt_center_new = np.delete(Xt_center, 0)
    Xt_left_new   = np.delete(Xt_left, 0)
    Xt_right_new  = np.delete(Xt_right, 0)
    steering_new  = np.delete(steering, 0)
    
    for index, item in enumerate(Xt_left_new):
        Xt_left_new[index] = item.strip()
        
    for index, item in enumerate(Xt_right_new):
        Xt_right_new[index] = item.strip()
    
    for index, item in enumerate(steering_new):
        steering_new[index] = float(item)        
    #print(steering_new[0])
        
    steering_new_left = []
    steering_new_left = np.copy(steering_new)
    for index, item in enumerate(steering_new_left):
        steering_new_left[index] = float(steering_new[index]) + float(0.125)
    #print(steering_new_left[0])

    steering_new_right = []
    steering_new_right = np.copy(steering_new)  
    for index, item in enumerate(steering_new_right):
        steering_new_right[index] = float(steering_new[index]) - float(0.125)
    #print(steering_new_right[0])      

    X_train = np.array(np.concatenate((Xt_center_new, Xt_left_new, Xt_right_new), axis=0))
    y_train = np.array(np.concatenate((steering_new, steering_new_left, steering_new_right), axis=0))
    
    print(y_train)
    print(data.shape)
    print(X_train.shape)
    print(y_train.shape)
    
    for index, item in enumerate(y_train):
        if (float(item) == 0.0):
            y_train = np.delete(y_train, [index], None)
            X_train = np.delete(X_train, [index], None)
               
    print(data.shape)
    print(X_train.shape)
    print(y_train.shape)

    # ***shuffle data 
    X_train, y_train = shuffle(X_train, y_train) 
    print(X_train.shape)
    print(y_train.shape)
    
    # ***split data
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.1, random_state=0)
    print(X_train.shape)
    print(y_train.shape)
    print(X_validation.shape)
    print(y_validation.shape)

    # plot steering angle
    #plt.figure(figsize=(6, 2))
    #plt.plot(y_train)
    #plt.show()

   
    # ***define the model
    
    model = Sequential()

    # deal with the color space    
    model.add(Convolution2D(3,1,1,border_mode='same',input_shape = (40,160,3), activation = 'elu'))

    # reduce to 20X80
    model.add(MaxPooling2D((2,2),strides=(2,2)))

    # two Convolution 32, 3X3
    model.add(Convolution2D(32, 3, 3,border_mode='same', activation='elu'))
    model.add(Convolution2D(32, 3, 3,border_mode='same', activation='elu'))

    # reduce to 10X40
    model.add(MaxPooling2D((2,2),strides=(2,2)))

    model.add(Dropout(0.5))

    # two Convolution 64, 3X3
    model.add(Convolution2D(64, 3, 3,border_mode='same', activation='elu'))
    model.add(Convolution2D(64, 3, 3,border_mode='same', activation='elu'))

    # reduce to 5X20
    model.add(MaxPooling2D((2,2),strides=(2,2)))

    model.add(Dropout(0.5))

    # two Convolution 128, 3X3
    model.add(Convolution2D(128, 3, 3,border_mode='same', activation='elu'))
    model.add(Convolution2D(128, 3, 3,border_mode='same', activation='elu'))

    # reduce to 2X10
    model.add(MaxPooling2D((2,2),strides=(2,2)))

    model.add(Dropout(0.5))

    # flatten
    model.add(Flatten(name='flatten'))

    # fully connected
    model.add(Dense(512, activation='elu'))
    model.add(Dense(128, activation='elu'))
    model.add(Dense(16, activation='elu'))
    model.add(Dense(1, activation='linear'))

    # Print out summary of the model
    model.summary()
    
    model.compile(optimizer=Adam(lr=FLAGS.lrate), loss='mse')
    
   
    # ***Train mode

    history = model.fit_generator(gen_batch(X_train, y_train, FLAGS.batch_size),
                                  16000,
                                  FLAGS.epochs,
                                  validation_data=gen_batch(X_validation, y_validation, FLAGS.batch_size),
                                  nb_val_samples=1600) 
 
    # ***Save model
    
    json = model.to_json()
    model.save('model.h5')
    with open('model.json', 'w') as f:
        f.write(json)
    
# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
        

    
