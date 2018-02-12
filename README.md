# Semantic Segmentation

## Goal

This project labels the pixels of a road in images using a Fully Convolutional Network (FCN).

---

## Example images classified

[//]: # (Image References)
[image1]: ./examples/sample1.png
[image2]: ./examples/sample2.png
[image3]: ./examples/sample3.png
[image4]: ./examples/sample4.png

![alt text][image1]

![alt text][image2]

![alt text][image3]

![alt text][image4]

All images classified from the test dataset can be seen [here](runs/1518933171.754386).

## Implementation Details

A modified version of the impressive VGG16 neural network image classification pipeline is used to classify road vs non-road pixels images. The program takes a pre-trained fully-convolutional network based on [FCN-8](https://github.com/shelhamer/fcn.berkeleyvision.org) and adds skip layers to allow for semantic segmentation of pixels in the image.

From the originally inputted layer, the input, keep probability, and layers 3, 4, and 7 are [extracted for further use](https://github.com/dinoboy197/CarND-Semantic-Segmentation/blob/master/main.py#L21-L44).

Next, [1x1 convolutions are constructed from layers 3, 4, and 7](https://github.com/dinoboy197/CarND-Semantic-Segmentation/blob/master/main.py#L65-L77) in an encoding step. Skip layers are inserted by adding the 1x1 convolutions from layers [3](https://github.com/dinoboy197/CarND-Semantic-Segmentation/blob/master/main.py#L84-L85) and [4](https://github.com/dinoboy197/CarND-Semantic-Segmentation/blob/master/main.py#L91-L92). Layers 3, 4, and 7 are [deconvolved in reverse order](https://github.com/dinoboy197/CarND-Semantic-Segmentation/blob/master/main.py#L79-L102) to complete the final piece of the decoding step.

After the network architecture is created, an [optimizer is created](https://github.com/dinoboy197/CarND-Semantic-Segmentation/blob/master/main.py#L109-L131) to minimize the softmax cross-entropy between the logits created by the network and the correct labels for image pixels. An optimizer using the Adam method is used to compute gradients for loss and gradients to variables in the networks.

The [neural network is then trained](https://github.com/dinoboy197/CarND-Semantic-Segmentation/blob/master/main.py#L136-L172) using a sample of labeled images for a maximum of 50 epochs. A mini-batch size of 10 images is used compromise between high memory footprint and smooth network convergence. The training step has an early terminator which does not continue to train the network if total training loss does not decrease for three subsequent epochs.

Finally, a separate sample of test images are run through the final neural network classifier for evaluation.

## Future improvements

* Use [the Cityscapes dataset](https://www.cityscapes-dataset.com/) for more images to train a network that can classify more than simply road / non-road pixels.
* Augment input images by flipping on the horizontal axis to improve network generalization.
* Apply trained classifer to a video stream for fun!

---

## Running the project

#### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

#### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

#### Run
Run the following command to run the project:
```
python main.py
```

#### Notes
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
