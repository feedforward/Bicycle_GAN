########################################
'''
Script for Testing the Bicyclic Gan 

Code Organizaiton :

Driver code
-----1. Collect Args 
-----2. Validate Args
-----3. Testing
----------1. Get Test Data
----------2. Define Model
----------3. Start a Session
----------4. Restore Pretrained weights
----------5. Start Summary writer 
----------6. Test 

Functions : 
-----1. validate_args
-----2. collect args
-----3. test

''' 
########################################

## Loading the dependences 
import argparse
import os
import tensorflow as tf

## Loading the modules from other files
from bcgan import Bicycle_GAN
from process_data import get_data

########################################
# Helper Functions for managing args ### 
########################################
def validate_args(args):
	''' Function for validating the arguments before running the 
	training code.'''

	datasets = ['maps', 'cityscapes', 'facades', 'edges2handbags',
				 'edges2shoes']
	if not(args.dataset in datasets):
		print('Invalid Dataset')
		exit(-1)

def collect_args():
	'''Function to collect the arguments from the terminal to run the
	training code.'''

	parser = argparse.ArgumentParser(description= 'To catch the arguments')
	parser.add_argument('--dataset', type= str, default= 'edges2shoes',
						help= 'Dataset name')
	parser.add_argument('--batch_size', type= int, default= 1,
						help= 'Batch size')
	parser.add_argument('--img_size', type= int, default= 256,
						help='Image size')
	parser.add_argument('--pretrained_weights', default= None,
						help= 'to train from the pretrained model')
	parser.add_argument('--gpu', type= str, default= '0',
						help= 'CUDA_VISIBLE_DEVICES')

	args = parser.parse_args()
	return args

########################################
######## Main function to train ######## 
########################################
def test(args):
	'''Function for testing the model,
	sets the gpu configrations,loads the test data, loads weights,
	 perfoms testing, writes summaries.'''

	## Setting the GPU configrations
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	
	## Limit from taking the whole gpu
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	## loading data 
	test_data = get_data(args.img_size, args.dataset, is_train= False, debug=False)
	
	## model definitoin
	with tf.variable_scope('bc_gan'):
		model = Bicycle_GAN(args)

	## Starting a session
	init = tf.global_variables_initializer()
	sess = tf.Session(config= config)
	sess.run(init)

	## savers and loaders
	global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= 'bc_gan')
	trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= 'bc_gan')
	loader = tf.train.Saver(global_vars)
	if args.pretrained_weights != None:
		loader.restore(sess, args.pretrained_weights)

	## results
	if not os.path.exists('./results'):
		os.mkdir('./results')
	write_dir = os.path.join('results')


	## Training
	model.test(sess, test_data, write_dir)

	print "Testing complete ...."

########################################
######## Driver code ################### 
########################################
if __name__ == "__main__":
	args = collect_args()
	print 'Colleted the Argumets'
	validate_args(args)
	test(args)