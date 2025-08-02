# Robotics with Neural Networks

* `implements the backbone of the resnet-50 neural network architecture`
* `3 convolutions are trained separate from the backbone for segmentation`
* `utilizes the CrossEntropy Loss Function which is advised with multiclass object designation`
* `utilizes ReLu() activation functions and the Adam weights optimizer`

## Neural Network Process

* 224 by 224 RGB images are passed into the backbone in batches of 4
* The Images are fed through the first layer of the CNN
* This layer extracts general features (contours, texture)
* The images are downsized through each convolution cycle
* In each convolutional layer the images are passed through a series of 3 operations
* First the images are passed through a Convolutional layer as tensors
* In the Convolution (typically) a 3x3 kernel calculates the dot product for each set of 3x3 pixels in the image
* the result is put onto an output 2D feature map
* In sum, its essentially a linear operation: output = W * input + b
* b is a bias that can be set but it isnt useful for our purposes
* W is the kernel which is also referred to as the weight
* Since there are typically 100s of kernels in a single convolution, all the output feature maps are stacked in a new tensor which is output to the BatchNorm function
* the BatchNorm function keeps all the tensor values within a range of (-1,1) with a standard deviation of 1
* Setting the standard deviation helps keep neurons alive as it offsets dot products that are 0 or very small negative values
* the ReLU function then looks at each value and sets the value to 0 if it is negative
* It also gives these values an associated gradient of 0
* The positive values are given a gradient of 1
* using this activation function helps delinearize the data and helps to reduce the effects of the vanishing gradient problem
* From there, the cycle repeats

## Segmentation

* when we want an output from the string of convolutions, we pass in the feature maps, and call for 3 output feature maps (3 kernels)
* these feature map values will correspond with class probabilities
* We dont need an activation function at this point because there will be no more future passes
* The backbone has downsized the image by a factor of 4 so we have to upsample to revert back to the original image shape
* The resulting tensor is finally passed as our returned value


## Backpropagation and Loss Function
* at the end of segmentation, the output Tensor is passed through a CrossEntropyLoss instance which outputs a scalar tensor that correlates with our error with respect to our true image (ground_truth)
* this scalar is used in our backward() function
* the backward function is the backpropagation process
* here the derivative of the linear string is calculated with the chain rule to determine how to optimize the input weights for each kernel (the elements)
* our optimization handler is through Adam (a program) with a set learning rate which sets a maximum on modification
* lastly, we call the optimizer to apply the new weights

## Visualizing the Output Mask

* We use argmax with our output tensor to generate the most likely class prediction for each pixel
* It compresses our tensor from 4d [batches, classes, height, width] to 3d [batches, height, width]
* These output images are the computer generated equivalents to our ground_truth masks

## Error

* Our error is collected from sklearns f1 score calculation w.r.t. our ground_truth images

## How I plan to use resnet50 with Robotics

* I plan to create a robot that takes in visual feedback to drive up and down my driveway
* At the end of the driveway, it'll pick up my mail and put it into a basket
* Lastly, it'll return up the hill to my house
* It'll use the resnet-50 object classification code for orientation
* I've already created a series of ground_truth images which I've edited through Photopea
* I took the photos on my Ipad and uploaded them to my laptop for exportation
* With minimal data (roughly 50 pictures), I've achieved a maximum f1 score of 0.91 with minimal overfitting
* Since my data is specialized to the features of my own driveway, I would have to collect and annotate much more data if i wanted to apply it broadly

## Running the Code

* You can edit the configs in the called machine learning function at the end of `image_processing.py`
* if you set patches to `True` then it'll separate each image into smaller patches (it'll take much longer and it's not advised for resnet-50)
* you can see a visual of the gt_prediction in `saved_images`