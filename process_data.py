########################################
'''
Module for loading the data  ... 

Pre- requisite - Data must be downloaded and stored in HDF5 format

Code Organizaiton :
------1. get data - Function to load the data.

''' 
########################################

## Loading the dependences 
import numpy as np
import os
import tqdm
import h5py
import time

########################################
########## Loading the data ############
########################################
def get_data(image_size=256, dataset='edges2shoes' , is_train=True, debug= False):
	'''function to get the training and validation data, dataset given as string,
		image size in int format, is_train in bool format for the train/valid data.'''
	## Check if the  data folder exists ... 
	assert os.path.exists('data'), 'Data Folder is not available.'
	#print dataset
	data_dir = os.path.join('data', dataset)
	## Check if the given dataset exists ...
	assert os.path.exists(data_dir), 'Dataset specified is not available.'
	## selecting between the train/val data ...
	if is_train ==  True :
		file_names = ['trainA', 'trainB']
	else :
		file_names = ['valA', 'valB']
	return_data = []
	## loading the images from hdf5 files
	for file_name in file_names:
		## load the hdf5 file
		curr_file = h5py.File(os.path.join(data_dir, '{}_{}.hy'.format(file_name, image_size)), 'r')    
		image_files = []
		## create a itearator
		iterations = tqdm.trange(0, len(curr_file), initial=0, total=len(curr_file))
		## iterating over the images ...
		for iteration_no in iterations:
			if iteration_no == 0:
				iterations.set_description('loading {}'.format(file_name))
			if iteration_no == 10 and debug == True:
				break  
			image_files.append(curr_file[str(iteration_no)]['image'].value.astype(np.float32))
		iterations.close()
		return_data.append(image_files)

	return return_data

if __name__ == '__main__':
	a = get_data()