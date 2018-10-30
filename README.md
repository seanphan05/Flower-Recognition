# Multi-class Flower Classification on Images using Deep Learning Convolutional Neural Network with Keras in R




**1.	Introduction:**

   Computer vision and deep learning field becomes more and more popular when it shows tremendous achievements in solving machine learning problems. With those advantages, automated image based classification makes classifying plants become easier and faster than ever.

   There are many parts of a plant that can be used to define and detect which species a plant should be listed into. However, the visually most prominent and perceivable part of a plant is its flowers, a subject of intense studies by botanists and often the key for species identification. Flowers usually exhibit great diversity in color, shape and texture, thus allowing to make use of a broad set of methods developed for object classification tasks.


**2.	Data Description:**

   Deep Learning with CNN technique is becoming a popular approach for classification problem in recent years. The main reason is because they can yield well performance with high significant accuracy compare to other traditional methods. This project will apply feature extracts and data augmentation to train the convolutional neural network model to classify images of flowers with their respective labels.

   The data collection is based on scraped data from flickr, google images, and yandex images. The pictures are divided into five classes: daisy, tulip, rose, sunflower, dandelion. For each class there are about 1,500 photos which are splitted out into 3 sets, 200 photos in testing set, 200 photos in validation set and 1,100 photos in training set. Photos are not high resolution, about 320x240 pixels and have different proportions in size. Data have been splitted in equal portion between sets thus makes this a balanced multi-­classification problem, which means classification accuracy will be an appropriate measure of success.

   The goal is to train a CNN model that would be able to classify flowers into five species which are same with their labels: daisy, dandelion, rose, sunflower and tulip.

**3.	Algorithms and Techniques**

**3.1		Layers :**


- Convolution: Convolution layers learn local patterns of the image such as edges, lines, colors and other visual elements. Two key characteristics of convolution network is that learned patterns are translation invariance which means they are learned only once time during the process, and patterns can be learned in spatial hierarchies which means one layer will learn a small local pattern, another will learn a larger local pattern and so on.

- Max-pooling: Max pooling layers work as filters to reduce the dimensionality of the images by setting a max value of channels for the output. Max pooling purpose to downsample the dense of feature map of images.


- Dropout: Dropout is a simple and effective technique for neural network in preventing overfitting when train the model. It works by randomly dropping out (setting to zero) a number of output features of the layer during training.


- Flatten: the output of the convolution layers will be reshaped before going through dense layers.


- Dense: dense layers are the connected networks that maps the scores of the convolutional layers into the correct labels with some activation function (softmax which is the best fit for multi-class classification is used)

**3.2	Activation functions:**

- ReLu Activation: ReLu (rectified linear unit) is a function meant to zero out negative values.


- Softmax Activation: Softmax function is applied to convert the output into probabilities with the sum of 1. For multi-class classification, Softmax function is the most appropriate.

**3.3	Data Augmentation:** 

   Data augmentation is a technique of generating more training data from existing training samples, by augmenting the samples via a number of random transformations such as rotation, shifting, shearing, zoom, flip, ect. CNN model generally perform better after augmentation thus it helps to mitigate overfitting.




**3.4	Feature extraction:**

   Feature extraction method uses the representations learned by a previous network to extract interesting features from new samples. These features are then run through a new classifier, which is trained from scratch. Applied feature extract can help cutting down training time and improve model performance.

**4.	Methodology:**

**4.1	Data Preprocessing:**

   Dataset had been download and arranged before applying any training process. Keras can generate training, validation, and testing data from defined directories in batches and process them with labels from those directories. This means, it can detect and label itself the data it is processing.

   Image_data_generator() function will be used with rescale option and data augmentation added. Only training data will be augmented. The train data then also shuffled during the training time while validation data was used in order to get the validation accuracy and validation loss during training.

   Feature extraction by extracting features in batches with data generated from image_data_generator() and flow_images_from_directory() functions and assign to new variables before feeding into model.

**4.2	Implementation:**

   In this project, I will process data classification in two difference approaches. 

- Building the model from scratch by stacking dense layers with appropriate number of layers chosen and number of hidden units for each layer. The model  will also be added with max-pooling layers, dropout, and data augmentation to prevent overfitting. The detail of this model is given below.

**Model**
<pre>
____________________________________________________________________________
Layer (type)                         Output Shape                   Param  #     
============================================================================
conv2d_1 (Conv2D)                    (None, 148, 148, 32)           896          
____________________________________________________________________________
Batch_normalization_1                (None, 148, 148, 32)           128          
____________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)       (None, 74, 74, 32)             0            
____________________________________________________________________________
conv2d_2 (Conv2D)                    (None, 72, 72, 64)             18496        
____________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)       (None, 36, 36, 64)             0            
____________________________________________________________________________
conv2d_3 (Conv2D)                    (None, 34, 34, 128)            73856        
____________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)       (None, 17, 17, 128)            0            
____________________________________________________________________________
conv2d_4 (Conv2D)                    (None, 15, 15, 128)            147584       
____________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)       (None, 7, 7, 128)              0            
____________________________________________________________________________
flatten_1 (Flatten)                  (None, 6272)                   0            
____________________________________________________________________________
dropout_1 (Dropout)                  (None, 6272)                   0            
____________________________________________________________________________
dense_1 (Dense)                      (None, 512)                    3211776      
____________________________________________________________________________
dense_2 (Dense)                      (None, 5)                      2565         
============================================================================
Total params: 3,455,301
Trainable params: 3,455,237
Non-trainable params: 64
____________________________________________________________________________
</pre>

- Building the model using VGG16 architecture from Keras. This pretrained convolutional base will be used in order to extract features from images. The convolutional base finally will be fed into a densely connected classifier with dropout layers added. Due to the limits of time and computational cost, it was impossible for me to apply data augmentation into this model. The detail of the architecture of the VGG16 convolutional base is given below.

**Model**
<pre>
____________________________________________________________________________
Layer (type)                           Output Shape                  Param #      
============================================================================
input_1 (InputLayer)                  (None, 150, 150, 3)            0            
____________________________________________________________________________
block1_conv1 (Conv2D)                 (None, 150, 150, 64)           1792         
____________________________________________________________________________
block1_conv2 (Conv2D)                 (None, 150, 150, 64)           36928        
____________________________________________________________________________
block1_pool (MaxPooling2D)            (None, 75, 75, 64)             0            
____________________________________________________________________________
block2_conv1 (Conv2D)                 (None, 75, 75, 128)            73856        
____________________________________________________________________________
block2_conv2 (Conv2D)                 (None, 75, 75, 128)            147584       
____________________________________________________________________________
block2_pool (MaxPooling2D)            (None, 37, 37, 128)            0            
____________________________________________________________________________
block3_conv1 (Conv2D)                 (None, 37, 37, 256)            295168       
____________________________________________________________________________
block3_conv2 (Conv2D)                 (None, 37, 37, 256)            590080       
____________________________________________________________________________
block3_conv3 (Conv2D)                 (None, 37, 37, 256)            590080       
____________________________________________________________________________
block3_pool (MaxPooling2D)            (None, 18, 18, 256)            0            
____________________________________________________________________________
block4_conv1 (Conv2D)                 (None, 18, 18, 512)            1180160      
____________________________________________________________________________
block4_conv2 (Conv2D)                 (None, 18, 18, 512)            2359808      
____________________________________________________________________________
block4_conv3 (Conv2D)                 (None, 18, 18, 512)            2359808      
____________________________________________________________________________
block4_pool (MaxPooling2D)            (None, 9, 9, 512)              0            
____________________________________________________________________________
block5_conv1 (Conv2D)                 (None, 9, 9, 512)              2359808      
____________________________________________________________________________
block5_conv2 (Conv2D)                 (None, 9, 9, 512)              2359808      
____________________________________________________________________________
block5_conv3 (Conv2D)                 (None, 9, 9, 512)              2359808      
____________________________________________________________________________
block5_pool (MaxPooling2D)            (None, 4, 4, 512)              0            
============================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
____________________________________________________________________________
</pre>

- Both approaches will have the same model configuration which are optimizer (optimizer_rmsprop), loss function (categorical_crossentropy), and metrics (accuracy).



**5.	Results**

**5.1	Model Evaluation**

   In first approach, during training time, I trained data with 100 epochs, It took around 4 hours to complete training process. The model returns validation loss at 0.8181 and validation accuracy of 69% while model accuracy is 68.6%. Look like the model seems not to be performing well. When I applied the model to classify test data, it show its terrible result with just only around 20% of images are correctly classified. After doing some tuning in parameters such as changing layers’ units, filters as well as learning rates in optimizer, the model has not significantly improved.

**Figure 1: Training and validation metrics using data augmentation**

![Alt text](https://github.com/seanphan05/Flower-Recognition/blob/master/Rplot01.png)

   In the second approach, I trained model with 30 epochs, each epoch took around 2 minutes. This approach is quite faster than the previous method, since the process only need to deal with 2 dense layers. More than that, the validation accuracy is 77.9% and the model accuracy is much better with 89.2%. After applying the model on testing data, I received the accuracy of 81.2% far better than the previous approach with just 20%.

**Figure 2: Training and validation metrics using pretrained base and feature extraction**

![Alt text](https://github.com/seanphan05/Flower-Recognition/blob/master/Rplot.png)

   Overall, the best model so far is using pretrained convolutional base from VGG16 architecture with dropout batch normalization to prevent overfitting. 

**5.2 	Visualization**

   To visualize what and how convolutional neural network learns representations in image classification, I implemented the two most useful and understandable techniques: Visualizing intermediate activation and Visualizing heatmaps of class activation. For each technique, I randomly picked one image from daisy test directory. 

**5.2.1	Visualization by visualizing intermediate activations**
	
   This technique particularly useful in explaining how inputs will be transformed on convnet layers and how individual filters works on convolutional neural network during training process. I used model from the first approach (model have been built from scratch with data augmentation added) to process the image. 

   The figure 3 below illustrate the original image compare to the second channel of the activation of the first layer of testing daisy image

**Figure 3: Original vs Second Channel of the Activation of the First Layer on the Test Daisy Picture**

![Alt text](https://github.com/seanphan05/Flower-Recognition/blob/master/daisy1.png)
![Alt text](https://github.com/seanphan05/Flower-Recognition/blob/master/daisy2.png)

**Figure 4: List of visualization for  every channel in every intermediate activation**

**Figure 4.1: Conv2d_3 (daisy_activations_3)**

![Alt text](https://github.com/seanphan05/Flower-Recognition/blob/master/daisy_activations_3_.png)
**Figure 4.2: Conv2d_5 (daisy_activation_5)**

![Alt text](https://github.com/seanphan05/Flower-Recognition/blob/master/daisy_activations_5_.png)
**Figure 4.3: Conv2d_8 (daisy_activation_8)**

![Alt text](https://github.com/seanphan05/Flower-Recognition/blob/master/daisy_activations_8_.png)	

	
   As we can see during the very beginning layers, the activations collected all information from image inputs, that is the reason why it is very visually interpretable. Going into deeper layers, the activations started obtaining more information related to class of the image such as “flower core”, “petal shape” or “petal patterns”. Thus, latest layers are less likely to carry the information of visual content of the image. The number of blank filters in following layers is increasing since the pattern encoded by the filter cannot be found in the input image. 

**5.2.2	Visualization by visualizing heatmaps of class activation**

   Visualizing heatmaps of class activation is very helpful when the users want to know which parts of the training image make the convnet classify that image into given classes. Moreover, object in image can also be able to located using this technique. This technique use VGG16 model to process a daisy image that I randomly picked from the testing data. 
   After loading network from VGG16 architecture and processing a random testing daisy image on it, the top three classes predicted for this image are given below:
  

| class_name   | class_description | score      |
| -------------|:-----------------:| ----------:|
|1  n11939491  | daisy             | 0.992138445|
|2  n02219486  | ant               | 0.002262975|
|3  n02206856  | bee               | 0.001018253|


   Unsurprisingly, the image had been classified as a daisy image with the highest score at 99.21%. there is 0.226% chance that the image will be classified as ant image, and 0.102% it is an bee image. Apparently, VGG16 network model had been done very well at its job to correctly classify this daisy image at a very high score.

   The next visualization is a comparison of original image with the one have superimposing the class activation heatmap on. This clearly shows that base on which part of the flower in image, the convnet make its final classification decision.
 
**Figure 5: Original vs Superimposing the Class Activation Heatmap on the Original Picture**

![Alt text](https://github.com/seanphan05/Flower-Recognition/blob/master/daisy3.png)
<br/>
![Alt text](https://github.com/seanphan05/Flower-Recognition/blob/master/daisy4.png)
<br/>

It is more likely that convnet classified the image as a daisy image based on the shape of petals around the flower core.



**Reference:**

Deep Learning with R - Francois Chollet and Joseph J. Allaire

Flower Images from Flower Recognition at Kaggle

Google image websites
	
