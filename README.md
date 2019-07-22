# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output/overview.png "Visualization"
[image2]: ./output/gray_out_image.png "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./output/extra-traffic-sign.png "Extra Traffic signs"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribute across each class. 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because with some experienment, the color information does not contribute significantly to the accuracy and intuitively people identify traffic sign based on shapes instead of color

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data in order to ensure numerical stability


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU                  |                                               |
| Convolution 3x3		| 1x1 stride, same padding, outputs 32*32*32	|
| RELU                  |                                               |
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Dropout               |  0.25                                         |
| Convolution 3x3	    | 1 * 1 stride, same padding, outputs 32X32X64  |
| RELU                  |                                               |
| Convolution 3x3	    | 1 * 1 stride, same padding, outputs 32X32X64  |
| RELU                  |                                               |
| Max pooling	      	| 2x2 stride,  outputs 8x8x64 				    |
| Dropout               |  0.25                                         |
| Fully connected		| 512        									|
| RELU                  |                                               |
| Dropout               |  0.25                                         |
| Fully connected		| 43        									|
| Softmax				|         							     		|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a batch size of 32 and used RMSprop as my optimizer, i trained for 10 epochs and apply a learning rate decay from 0.0001 with decay=1e-6

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.977
* validation set accuracy of 0.955
* test set accuracy of 0.95

For the experienment, i chose the well-know alex net, alex net is known for its good results on image net dataset on classification, given the similarity of task, i decided to chose this architecture as my first try and it worked pretty well.
After 10 epochs, The training accuracy and validation accuracy yield monotomic decrease. And the test accuracy is 95% which is good enough for my purpose
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 8 German traffic signs that I found on the web:

![alt text][image4] 
Four of them are difficult to classify because the source includes embient and the resized version distorted the information. The reason i picked them originally is to test the translation invariant property of CNN, however, i realized that i overlooked the fact the the resize distort the image, ideally a better data would be to get a focused view on the traffic sign through object detection. But that is a different class of problem.

The other four images that came from a focused view yields much better result. 
#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| work ahead     		| Wild animals crossing 						|
| Stop Sign      		| Stop sign   									| 
| Yield					| Yield											|
| traffic signal	    | traffic signal				 				|
| no passing			| End of no passing     		    			|


The model was able to correctly guess 4 of the 8 traffic signs, This seemingly low accuracy is due to the fact that i purposely did not use images from the same source and the images i used are not focused enough and the distorsion introduced by resizing the image make the prediction harder.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively unsure that this is a Wild animals crossing  (probability of 0.6), with 30% probability of classifying it correctly. I believe the error is due to the distorsion

#### Correct: work ahead 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Wild animals crossing   					    | 
| .30     				| Road work 									|
| .06					| Keep left										|
| .005	      			| Speed limit (80km/h)					 		|
| .004				    | No passing for vehicles over 3.5 metric tons  |

#### Correct Stop sign
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .987         			| stop sign  					                | 
The probability is overwhelmingly large the rest are just noise

#### Correct: Yield
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .987         			| yield      					                | 

#### Correct: traffic signal
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .522         			|traffic signal     					        | 
| .477         			| General caution      					        | 

This is particulary interesting, as the shape of traffic signal is very similar to that of general causion. That is probably the reason for the close probability. This also shows that potentially the color might be useful in this case

#### no passing
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .714         			| end of no passing     					        | 
| .285         			| no passing        					        | 


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


