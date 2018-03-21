---
layout: post-wo-sidebar
title: Toward Multimodal Image-to-Image Translation
date: 2018-03-21 15:47:20 +0300
description: 
img: teaser.jpg
content_type: gan
tags: [GAN, Video Generation]
---

## Code Walkthrough for Bi-Cycle GAN  


## List of files 

1. process_data.py
2. bcgan.py
3. networks.py
4. layers.py
5. test.py
6. train.py

![Network]({{site.baseurl}}/assets/img/adfasdfads.gif)
## Libraries
```
import tensorflow as tf
```
Code is written in python using tensorflow library.

Other dependencies
numpy, scipy, os, argparse, tqdm, h5py, time, random.
################################################################################################################




## process_data.py [Data preprocessing]

* modules : get_data

1. downloaded data is loaded using this module
2. augmentation is done on the fly - not part of this step.

```
def get_data(image_size=256, dataset='edges2shoes' , is_train=True, debug= False):
	'''function to get the training and validation data, dataset given as string,
		image size in int format, is_train in bool format for the train/valid data.'''
		.
		.
		.
		.
	return return_data
```
![Network]({{site.baseurl}}/assets/img/screenshot.png)
################################################################################################################




## bcgan.py [BC-GAN Model definitions]

* modules : Bicycle_GAN
* dependencies 
```
from network import generator, discriminator, encoder
```

### module : Bicycle_GAN
module is a class definition - for bicyclic gan

#### functions:
1. constructor
2. summary_create
3. train
4. test

#### function: constructor
1. creates all the necessary variable in the class object.
2. uses the modules generator, discriminator and encoder to create cVAE-GAN and cLR-GAN
3. formulates the loss functions
4. optimizers
5. update ops(taking care of batchnorm updates)

#### function: summary_create
1. create the tensorboard summaries for all costs, and images
2. merging all summaries 

#### function: train
1. runs the main training loop
2. loss minimization and gradient updates 
3. learning rate is periodically decayed 
4. summaries are periodically written

#### function: test
1. loads the pretrained weights
2. generates the images by random sampling 
3. saves the images 

```
class Bicycle_GAN(object):
	def __init__(self, ...):
		.
		.
		return xxx
	def summary_create(self):
		.
		.

	def train(self, sess, data, saver, summary_writer):
		.
		.

	def test(self, sess, data, write_dir):
		.
		.


```
![Network]({{site.baseurl}}/assets/img/paper-figure.png)
################################################################################################################




## network.py [GEN, DISC, ENC Model definitions]
* modules - generator, discriminator, encoder
* dependencies 
```
from layers import * ( wrapper functions for all the layers)
```

### module : generator

* for creating the generator graph definition, with all the conv layers, normalizations, and activations.
* returns the final layer output 

### module : discriminator

* for creating the discriminator graph definition, uses the deconv layers in addition to other layers to increase the spatial size.
* returns the final layer output 



### module : encoder

* for creating the encoder graph definition, uses the residual skip connections along with other layers.
* returns the final layer output 

```
class Generator(object):
	def __init__():
		.
		.
	def__call__():
		.
		.

class Discriminator(object):
	def __init__():
		.
		.
	def__call__():
		.
		.

class Encoder(object):
	def __init__():
		.
		.
	def__call__():
		.
		.

```

################################################################################################################



## layers.py [Wrappers for tf.layers]
* modules - conv2d, flatten, residual etc …

wrapper functions on top of the tensorflow implementations of the defined layers.
```
def normalization(input, is_train, norm=None):
	.
	.
	return output

def conv2d(input, is_train, norm=None):
	.
	.
	return output

def residual(input, is_train, norm=None):
	.
	.
	return output

```
################################################################################################################


## train.py
* modules - collect_args, validate_args, train
* dependencies - Bicycle_GAN, get_data

### function: collect_args
* collect the model parameters and training parameters using the argparse 

### function: validate_args
* validates the collected arguments are allowable values

### function: train

1. sets up the GPU environment and variables
2. loads the training data
3. creates the BiCycle GAN model definition
4. load the pretrained weights if exists
5. call the training function in Bicycle_GAN

```
def validate_args(args):
	"""Validating the arguments"""
	.
	.

def collect_args():
	"""Collecting the arguments"""
	.
	.

def train(args):
	"""Training the Model"""


if __name__ == "__main__":
	args = collect_args()
	print 'Colleted the Argumets'
	validate_args(args)
	train(args)

```
################################################################################################################



## test.py
* modules - collect_args, validate_args, train
* dependencies - Bicycle_GAN, get_data

### function: collect_args
* collect the model parameters and training parameters using the argparse 

### function: validate_args
* validates the collected arguments are allowable values

### function: test
1. sets up the GPU environment and variables
2. loads the testing data
3. creates the BiCycle GAN model definition
4. load the pretrained weights
5. call the test function in BiCycle_GAN

```
def validate_args(args):
	"""Validating the arguments"""
	.
	.

def collect_args():
	"""Collecting the arguments"""
	.
	.

def test(args):
	"""Training the Model"""


if __name__ == "__main__":
	args = collect_args()
	print 'Colleted the Argumets'
	validate_args(args)
	test(args)

```
################################################################################################################


## Usage 
* Training - default [edges2shoes, size=256] 
```
python train.py --dataset edges2shoes --batch_size 1 --img_size 256 --gpu 1
```
* Testing
```
python test.py --pretrained_weights 'weights/location/go/here'
```
* Tensorboard 
```
tensorboard --logdir=./logs
```
* in browser - localhost:6006/


## Authors
Jun-Yan Zhu, Richard Zhang, Deepak Pathak, Trevor Darrell, Alexei A. Efros, Oliver Wang, Eli Shechtman

## Sources
[Paper](https://arxiv.org/abs/1711.11586)

