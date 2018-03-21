########################################
'''
Bicycle GAN - Model Definition

Code Organizaiton :

Bicycle_GAN - Class Definitoin
-----1. constructor 
-----2. summary_create
-----3. train
-----4. test

'''
########################################

## Importing libraries 
import numpy as np
import tensorflow as tf
import os
import tqdm
import random
from scipy.misc import imsave
## Importing Modules from other files
from networks import Generator, Discriminator, Encoder

########################################
######## Model DEF - Bicycle GAN ####### 
########################################

class Bicycle_GAN(object):
	def __init__(self, args, lr=0.1, latent_dim=8, lambda_latent=0.5,
					lambda_kl= 0.001, lambda_recon= 10, is_train = True,  ):
		## Parameters 
		self.batch_size = args.batch_size
		self.latent_dim = latent_dim
		self.image_size = args.img_size
		self.lambda_kl = lambda_kl 
		self.lambda_recon = lambda_recon
		self.lambda_latent = lambda_latent
		self.is_train = tf.placeholder(tf.bool, name= 'is_training')
		self.lr = tf.placeholder(tf.float32, name='learning_rate')
		self.A = tf.placeholder(tf.float32, [self.batch_size, self.image_size,
								self.image_size, 3], name= 'A') 
		self.B = tf.placeholder(tf.float32, [self.batch_size, self.image_size,
								self.image_size, 3], name= 'B')
		self.z = tf.placeholder(tf.float32, [self.batch_size, self.latent_dim], 
								name= 'z')

		## Augmentation
		def aug_img(image):
			aug_strength = 30
			aug_size = self.image_size + aug_strength
			image_resized = tf.image.resize_images(image, [aug_size, aug_size])
			image_cropped = tf.random_crop(image_resized, [self.batch_size, self.image_size,
								self.image_size, 3])
			## work-around as tf-flip doesn't support 4D-batch
			image_flipped = tf.map_fn(lambda image_iter: tf.image.random_flip_left_right(image_iter), image_cropped)
			return image_flipped
		A = tf.cond(self.is_train,
					 lambda: aug_img(self.A), lambda: self.A)
		B = tf.cond(self.is_train, 
					lambda: aug_img(self.B), lambda: self.B)
		## Generator
		with tf.variable_scope('generator'):
			Gen = Generator(self.image_size, self.is_train)

		## Discriminator
		with tf.variable_scope('discriminator'):
			Disc = Discriminator(self.image_size, self.is_train)

		## Encoder
		with tf.variable_scope('encoder'):
			Enc = Encoder(self.image_size, self.is_train, self.latent_dim)

		## cVAE-GAN
		with tf.variable_scope('encoder'):
			z_enc, z_enc_mu, z_enc_log_sigma = Enc(B)
		
		with tf.variable_scope('generator'):
			self.B_hat_enc = Gen(A, z_enc)

		## cLR-GAN 
		with tf.variable_scope('generator', reuse=True):
			self.B_hat = Gen(A, self.z)
		with tf.variable_scope('encoder', reuse= True):
			z_hat, z_hat_mu, z_hat_log_sigma = Enc(self.B_hat)

		## Disc
		with tf.variable_scope('discriminator'):
			self.real = Disc(B)
		with tf.variable_scope('discriminator', reuse=True):
			self.fake = Disc(self.B_hat)
			self.fake_enc = Disc(self.B_hat_enc)

		## losses
		self.vae_gan_cost = tf.reduce_mean(tf.squared_difference(self.real, 0.9)) + \
						tf.reduce_mean(tf.square(self.fake_enc))
		self.recon_img_cost = tf.reduce_mean(tf.abs(B - self.B_hat_enc))
		self.gan_cost = tf.reduce_mean(tf.squared_difference(self.real, 0.9)) + \
					tf.reduce_mean(tf.square(self.fake))
		self.recon_latent_cost = tf.reduce_mean(tf.abs(self.z-z_hat))
		self.kl_div_cost =  -0.5*tf.reduce_mean(1 + 2*z_enc_log_sigma - z_enc_mu**2 -\
							tf.exp(2* z_enc_log_sigma))
		self.vec_cost = [self.vae_gan_cost, self.recon_img_cost, self.gan_cost, self.recon_latent_cost, 
						self.kl_div_cost]
		weight_vec = [1, -self.lambda_recon, 1, -self.lambda_latent, self.lambda_kl]

		self.cost = tf.reduce_sum([self.vec_cost[i]* weight_vec[i] for i in range(len(self.vec_cost)) ])

		## Optimizers
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		self.optim_gen = tf.train.AdamOptimizer(self.lr, beta1=0.5)
		self.optim_disc = tf.train.AdamOptimizer(self.lr, beta1=0.5)
		self.optim_enc = tf.train.AdamOptimizer(self.lr, beta1=0.5)
		
		## Collecting the trainalbe variables
		gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='bc_gan/generator')
		disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='bc_gan/discriminator')
		enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='bc_gan/encoder')

		## Defining the training operation 
		with tf.control_dependencies(update_ops):
			self.train_op_gen = self.optim_gen.minimize(-self.cost, var_list=gen_vars)
			self.train_op_disc = self.optim_disc.minimize(self.cost, var_list= disc_vars)
			self.train_op_enc = self.optim_enc.minimize(-self.cost, var_list=enc_vars)

		## Joing the training ops 
		self.train_ops = [self.train_op_gen, self.train_op_disc, self.train_op_enc]
		## Summary Create
		def summary_create(self):
			## Image summaries
			tf.summary.image('A', self.A[0:1])
			tf.summary.image('B', self.B[0:1])
			tf.summary.image('B^', self.B_hat[0:1])
			tf.summary.image('B^-enc', self.B_hat_enc[0:1])
			## GEN - DISC summaries - min max game  
			tf.summary.scalar('fake', tf.reduce_mean(self.fake))
			tf.summary.scalar('fake_enc', tf.reduce_mean(self.fake_enc))
			tf.summary.scalar('real', tf.reduce_mean(self.real))
			tf.summary.scalar('learning_rate', self.lr)
			## cost summaries		
			tf.summary.scalar('cost_vae_gan', self.vae_gan_cost)
			tf.summary.scalar('cost_recon_img', self.recon_img_cost)
			tf.summary.scalar('cost_gan_cost', self.gan_cost)
			tf.summary.scalar('cost_recon_latent', self.recon_latent_cost)
			tf.summary.scalar('cost_kl_div', self.kl_div_cost)
			tf.summary.scalar('cost_final', self.cost)
			## Merge Summaries
			self.merge_op = tf.summary.merge_all()

		summary_create(self)
	


	def train(self, sess, data, saver, summary_writer):
		## Data validation
		assert len(data) == 2 , 'Invalid data A and B not given. '
		assert len(data[0]) == len(data[1]), 'A and B - nonequal no of samples.'
		if not os.path.exists('images'):
			os.makedirs('images')
		if not os.path.exists('weights'):
			os.makedirs('weights')
		## to write summaries 
		display_step = 100

		## adjusting learning rates 
		lr_init = 5e-4
		lr_decay = lr_init/2
		lr_curr = lr_init 

		## adjusting iterations and epocha
		no_samples = len(data[0])
		no_batches = no_samples//self.batch_size 
		no_samples_per_epoch = no_batches * self.batch_size
		iterations = tqdm.trange(0, no_samples_per_epoch*10, total= no_samples_per_epoch*10,
									initial= 0)

		## Main training loop
		for iteration_no in iterations:
			## setting trainin description
			iterations.set_description('training')
			curr_epoch = iteration_no//no_samples_per_epoch
			iteration_no_curr_epoch = iteration_no - curr_epoch* no_samples_per_epoch
			## Adjsuting learning rate
			if curr_epoch > 8:
				lr_curr = lr_init - (curr_epoch-8)* lr_decay 
			## shuffling the data after each epoch
			if iteration_no_curr_epoch == 0:
				data_zip = zip(data[0], data[1])
				random.shuffle(data_zip)

				data[0], data[1] = zip(*data_zip)
			## collect the training batch
			train_A = data[0][iteration_no_curr_epoch*self.batch_size: (iteration_no_curr_epoch+1)*self.batch_size] 
			train_B = data[1][iteration_no_curr_epoch*self.batch_size: (iteration_no_curr_epoch+1)*self.batch_size] 
			train_z = np.random.normal(size=(self.batch_size, self.latent_dim))

			##  running session - with or without summary selection 
			train_outs = self.train_ops + [ self.cost]
			if iteration_no% display_step == 0:
				train_outs = self.train_ops + [self.merge_op]

			## input to be feed for each iteration
			feed_dict = {self.A:train_A, self.B: train_B,
							self.z: train_z, self.lr: lr_curr,
							self.is_train: True}
			## executing the backprop
			train_outputs = sess.run(train_outs, feed_dict= feed_dict)

			## Writing the summaries
			if iteration_no % display_step == 0:
				feed_dict = {self.A:train_A, self.B: train_B,
								self.z: train_z, self.lr: lr_curr,
								self.is_train: False}
				train_B_hat = sess.run(self.B_hat, feed_dict)
				train_summary = sess.run(self.merge_op, feed_dict)
				summary_writer.add_summary(train_summary, iteration_no)
				if not os.path.exists('images'):
					os.makedirs('images')
				train_B_hat_normalized = (train_B_hat - np.min(train_B_hat))/(np.max(train_B_hat))
				imsave('images/b_hat_{}.png'.format(iteration_no), np.squeeze(train_B_hat_normalized, axis= 0))
				saver.save(sess, 'weights/iteration',global_step = iteration_no,  write_meta_graph = False)

	def test(self, sess, data, write_dir):
		## checking for corrects of data
		assert len(data) ==2, "Invalid data"
		assert len(data[0]) == len(data[1]), "Mismatch - Number of samples in A and B"
		no_test_samples = len(data[0])
		## iterator over the test samples
		iter_test_samples = tqdm.trange(0, len(data[0]), initial=0, total= len(data[0]))
		## Main testing loop
		for iteration_no in iter_test_samples:
			## setting description
			iter_test_samples.set_description('test data - evaluation')
			## data collection
			test_A = data[0][iteration_no*self.batch_size:(iteration_no+1)*self.batch_size]
			test_B = data[1][iteration_no*self.batch_size: (iteration_no+1)*self.batch_size]

			## To store the images samples 
			test_rand_samples = []
			test_lin_samples = []

			## random sampling collector
			test_rand_samples.append(test_A)
			test_rand_samples.append(test_B)
			## Linear sampling collector
			test_lin_samples.append(test_A)
			test_lin_samples.append(test_B)

			image_grid_rand = []
			image_grid_lin = []

			## image sapce sampling 
			for i in range(28):
				test_z = np.random.normal(size=(1, self.latent_dim))
				feed_dict ={self.A : test_A, self.z :test_z, self.is_train: False}
				test_B_hat = sess.run(self.B_hat,feed_dict= feed_dict)
				test_rand_samples.append(test_B_hat)

				z = np.zeros((1, self.latent_dim))
				z[0][0] = (i/28.0 - 0.5)* 2.0
				feed_dict ={self.A : test_A, self.z :z, self.is_train: False}
				test_B_hat = sess.run(self.B_hat, feed_dict = feed_dict)
				test_lin_samples.append(test_B_hat)

			## Storing the images in a grid format
			for i in range(6):
				curr_row = np.concatenate(test_rand_samples[i*5:(i+1)*5], axis=2)
				image_grid_rand.append(curr_row)
			image_grid = np.concatenate(image_grid_rand, axis=1)
			image_grid = np.squeeze(image_grid, axis=0)
			curr_file_name = os.path.join(write_dir, 'B_hat_rand_{}.png'.format(iteration_no))
			imsave(curr_file_name, image_grid)
			
			## stroring the images in a grid format
			for i in range(6):
				curr_row = np.concatenate(test_lin_samples[i*5:(i+1)*5], axis=2)
				image_grid_lin.append(curr_row)
			image_grid = np.concatenate(image_grid_lin, axis=1)
			image_grid = np.squeeze(image_grid, axis=0)
			curr_file_name = os.path.join(write_dir, 'B_hat_lin_{}.png'.format(iteration_no))
			imsave(curr_file_name, image_grid)
			








