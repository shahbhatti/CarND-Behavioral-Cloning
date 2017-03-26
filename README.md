**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results
* I have added a video of myrun (shahzrun) and it is also posted on youtube (https://youtu.be/gPbXwaDHjIU)

#### 2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

python drive.py model.h5

#### 3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

#### 1. An appropriate model arcthiecture has been employed

Below is a summary of my model. It consists of a convolution neural network with 3x3 filter sizes and depths of 2X32, 2X64 and 2X128. It includes MaxPooling and Dropouts. The fully connected layers are added after the flattening layer.

The model includes ELU layers to introduce nonlinearity, and the data is normalized and re-sized in preprocessing.

____________________________________________________________________________________________________
Layer, (type), (Output Shape), Num, Connected-to                     
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 40, 160, 3)    12          convolution2d_input_1[0][0]      
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 20, 80, 3)     0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 20, 80, 32)    896         maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 20, 80, 32)    9248        convolution2d_2[0][0]            
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 10, 40, 32)    0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 10, 40, 32)    0           maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 10, 40, 64)    18496       dropout_1[0][0]                  
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 10, 40, 64)    36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 5, 20, 64)     0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 5, 20, 64)     0           maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
convolution2d_6 (Convolution2D)  (None, 5, 20, 128)    73856       dropout_2[0][0]                  
____________________________________________________________________________________________________
convolution2d_7 (Convolution2D)  (None, 5, 20, 128)    147584      convolution2d_6[0][0]            
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 2, 10, 128)    0           convolution2d_7[0][0]            
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 2, 10, 128)    0           maxpooling2d_4[0][0]             
____________________________________________________________________________________________________
flatten (Flatten)                (None, 2560)          0           dropout_3[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 512)           1311232     flatten[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 128)           65664       dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 16)            2064        dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             17          dense_3[0][0]                    
____________________________________________________________________________________________________
Total params: 1,665,997
Trainable params: 1,665,997
Non-trainable params: 0
____________________________________________________________________________________________________
Epoch 1/10

tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties:
name: GRID K520
major: 3 minor: 0 memoryClockRate (GHz) 0.797
pciBusID 0000:00:03.0
Total memory: 3.94GiB
Free memory: 3.91GiB
tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0
tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y
tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GRID K520, pci bus id: 0000:00:03.0)

16000/16000 [==============================] - 38s - loss: 0.0254 - val_loss: 0.0134

Epoch 2/10

16000/16000 [==============================] - 35s - loss: 0.0151 - val_loss: 0.0111

Epoch 3/10

16000/16000 [==============================] - 35s - loss: 0.0132 - val_loss: 0.0118

Epoch 4/10

16000/16000 [==============================] - 36s - loss: 0.0129 - val_loss: 0.0124

Epoch 5/10

16000/16000 [==============================] - 36s - loss: 0.0125 - val_loss: 0.0113

Epoch 6/10

16000/16000 [==============================] - 36s - loss: 0.0122 - val_loss: 0.0121

Epoch 7/10

16000/16000 [==============================] - 35s - loss: 0.0118 - val_loss: 0.0119

Epoch 8/10

16000/16000 [==============================] - 35s - loss: 0.0128 - val_loss: 0.0117

Epoch 9/10

16000/16000
[==============================] - 35s - loss: 0.0118 - val_loss: 0.0104

Epoch 10/10

16000/16000 [==============================] - 35s - loss: 0.0119 - val_loss: 0.0110


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the data provided by udacity. I used a combination of center lane driving, recovering from the left and right sides of the road. I adjusted the driving angles for the left and right side images. I also flipped the images to increase the size of training set.

## Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture based on the Nvidia model and literature search (http://cs231n.github.io/convolutional-networks/, for example). I choose this approach becasue the model has been publiched and has proven results.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. I added the dropout layers to reduce overfitting.

I also rezized the image from 160X360X3 to 40X120X to improve memory utilization and increase performance. I added Pooling layers to perform a downsampling operation along the spatial dimensions (width, height), resulting in volume such as [10x40x32].

The final step was to run the simulator to see how well the car was driving around track one. After many, many, many tries I was able keep on the road! I experimented with a lot of the hyper parameters, such as the learning rate and batch size. I had to make sure my preprocessing in model.py matched the preporcessing drive.py (That cost me a day!).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture is summarized above using the model.summary() function.

I have not included a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric), but it looks very much like the covnet described in literature ((http://cs231n.github.io/convolutional-networks/).

#### 3. Creation of the Training Set & Training Process

I used the training data provided by udacity. I did split the training data into training and validation data.

I finally randomly shuffled the data set and put 10% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the loss, that is, it stopped reducing the loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
