########################################
'''
Wrapper functions for the layers in the network. 

Code Organizaiton :

Modules 
-----1.  
-----2. 
-----3. 
''' 
########################################

## Loading the dependences 
import numpy as np
import tensorflow as tf

########################################
###### Normalization function ########## 
########################################
def normalization(input, is_train, norm=None):
	if norm == 'batch':
		with tf.variable_scope('batch_norm', reuse=False):
			output = tf.contrib.layers.batch_norm(input, decay=0.99,
												center=True, scale=True,
												is_training=True)
	else:
		return input
	return output

########################################
#### Non linear activation function #### 
########################################
def activation_function(input, activation=None):
	if activation == 'relu':
		return tf.nn.relu(input)
	elif activation == 'leaky':
		return tf.contrib.keras.layers.LeakyReLU(0.2)(input)
	elif activation == 'tanh':
		return tf.tanh(input)
	else:
		return input

########################################
######### Flattening layer ############# 
########################################
def flatten(input):
	return tf.reshape(input, [-1, np.prod(input.get_shape().as_list()[1:])])

########################################
###### Convolutional layer ############# 
########################################
def conv2d(input, no_fil, fil_size, stride, pad='SAME', bias=False):
	str_shape = [1, stride, stride, 1]
	fil_shape = [fil_size, fil_size, input.get_shape()[3], no_fil]

	weights = tf.get_variable('weights', fil_shape, tf.float32, tf.random_normal_initializer(0.0,0.2) )

	conv = tf.nn.conv2d(input, weights, str_shape, padding='SAME')
	if bias == True:
		bias = tf.get_variable('bias', [1,1,1,no_fil], initializer= tf.constant_initializer(0.0))
		conv += bias
	return conv

########################################
### Transpose Convolutional layer ###### 
########################################
def conv2d_transpose(input, no_fil, fil_size, stride, pad='SAME'):
	in_no_fil, in_height, in_width, in_channels = input.get_shape().as_list()
	str_shape = [1, stride, stride, 1]
	fil_shape = [fil_size, fil_size, no_fil, in_channels]
	output_shape = [in_no_fil, in_height* stride, in_width*stride, no_fil]

	weights = tf.get_variable('weights', fil_shape, tf.float32, tf.random_normal_initializer(0.0, 0.02))
	deconv = tf.nn.conv2d_transpose(input, weights, output_shape, str_shape, pad)
	return deconv

########################################
###### Fully connected layer ########### 
########################################
def fc(input, out_dim, name, is_train, norm=None, activation=None):
	batch_size, no_units = input.get_shape()
	with tf.variable_scope(name, reuse=False):
		weights = tf.get_variable('weights', [no_units, out_dim], tf.float32, tf.random_normal_initializer(0.0, 0.02))
		output = tf.matmul(input, weights)
		bias = tf.get_variable('bias', [out_dim], initializer=tf.constant_initializer(0.0))
		output = output + bias
		output = normalization(activation_function(output, None), is_train, norm)
		return output

########################################
#### Conv block - conv + norm + act #### 
########################################
def conv_block(input, no_fil, name, ker_size, stride, is_train, norm,
		  activation, pad='SAME', bias=False):
	with tf.variable_scope(name, reuse=False):
		output = conv2d(input, no_fil, ker_size, stride, pad, bias=bias)
		output = normalization(output, is_train, norm)
		output = activation_function(output, activation)
		return output

########################################
######### Residual block  ############## 
########################################
def residual(input, no_fil, name, is_train, norm, pad='REFLECT',
			 bias=False):
	with tf.variable_scope(name, reuse=False):
		## First sub-block in the res-block
		with tf.variable_scope('res_1', reuse=False):
			output = conv2d(input, no_fil, 3, 1, pad, bias=bias)
			output = normalization(output, is_train, norm)
			output = tf.nn.relu(output)
		## Second sub-block in the res-block
		with tf.variable_scope('res_2', reuse=False):
			output = conv2d(output, no_fil, 3, 1, pad, bias=bias)
			output = normalization(output, is_train, norm)
		## Adding the skip connection 
		with tf.variable_scope('skip_conn', reuse=False):
			skip_conn = conv2d(input, no_fil, 1, 1, pad, bias=bias)
		return tf.nn.relu(output + skip_conn)

########################################
###### Deconvolution block ############# 
########################################
def deconv_block(input, no_fil, name, ker_size, stride, is_train,
				 norm, activation):
	with tf.variable_scope(name, reuse=False):
		output = conv2d_transpose(input, no_fil, ker_size, stride)
		output = normalization(output, is_train, norm)
		output = activation_function(output, activation)
		return output