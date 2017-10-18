#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization_train.jpg "Visualization Train"
[image3]: ./examples/visualization_test.jpg "Visualization Test"
[image2]: ./examples/visualization_validation.jpg "Visualization Validation"
[image4]: ./examples/grayscale.jpg "GrayScale"
[image5]: ./web-data/30kph.jpg "30 kmp"
[image6]: ./web-data/children-crossing.jpg "Child Crossing"
[image7]: ./web-data/keep_left.jpg "Keep Left"
[image8]: ./web-data/Right-of-way.jpg "Right of way"
[image9]: ./web-data/Roadworks.jpg "Road Works"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/vipinn123/Traffic_sign_classifier)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.


Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across various classes in the train, validation and test set

![alt text][image1]
![alt text][image2]
![alt text][image3]

The frequency distribution indicates that the data is skewed towards certain classes. Images of speed limits 30 and 50 are high. So is the case with yield and Priority road. The speed limit 20 has lowest sample size.

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because that reduces the size of my feature vector. It also equilizes variations in colors. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image4]

I normalized the image data using the mean and the standard deviation of the grayscale pixel range from training set. 

I decided to try out the model with the given data set first, before I tried any augmentation. The given data set itself provided me with reasonably good results. Hence i did not feel the need to augment the data set with additional data. However if required we could augment the data by collecting more samples from the web or by rotating and flipping the existing images. We need to be cautious while flipping images because certain images when flipped, would belong to a completely different class (e.g. keep left vs keep right)

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale, normalized image   		| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x10 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x10 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x30	|
| RELU      			|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x30 					|
| Fully connected		| Input 750, Output 300        					|
| Fully connected		| Input 300, Output 120        					|
| Fully connected		| Input 120, Output 43        					|
-------------------------------------------------------------------------

 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a variation of LeNet. I used the AdamOptimizer. The batch size is 128. I ran the model for 50 epochs. Learning rate was set to 0.001. The keep probability was kept constant at 0.8 for all fully connected networks. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of greater than 93%  
* test set accuracy of greater than 93%


If a well known architecture was chosen:
* What architecture was chosen : LeNet is the architecture I chose
* Why did you believe it would be relevant to the traffic sign application: LeNet has proven capabilites for similar image classification
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? The model very quickly converged and was consistent in validation and test data.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)	| Speed limit (30km/h)							| 
| Children crossing		| Children crossing								|
| Keep Left				| Keep Left										|
| Right-of-way     		| Right-of-way					 				|
| Road works			| Road works        							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 47th cell of the Ipython notebook.

The code was able to able to predict the category of the images with a probability of 1.0 for speed limit (30km/h), keep left and Right of way. For the children crossing sign we get a top probability of 0.99. For Roadworks we get a probability of 0.98.

