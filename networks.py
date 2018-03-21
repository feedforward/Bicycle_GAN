import tensorflow as tf
from layers import *

class Generator(object):
	"""Generator architecture """
	def __init__(self, image_size, is_train):
		self.image_size = image_size
		self.is_train = is_train

	def __call__(self, input, z):
		## Number of filters in each layer
		Encoder_filters = [64, 128, 256, 512, 512, 512, 512, 512]
		Decoder_filters = [Encoder_filters[len(Encoder_filters) -1 -i] \
							for i in range(len(Encoder_filters)) ][1:]
		batch_size = int(input.get_shape()[0])       #.as_list[0]
		latent_dim = int(z.get_shape()[-1])      #.as_list[-1]
		## spatial replication of the latent vector
		z = tf.reshape(z, [batch_size,1,1,latent_dim])
		z = tf.tile(z, [1,self.image_size, self.image_size, 1])

		## for the collection of activations at all layers 
		end_points = {}
		end_points_names = []
		layer_name = 'init' 
		end_points[layer_name] =  tf.concat([input, z], axis=3)
		
		## forming the layers in the encoder 
		for i in range(len(Encoder_filters)):
			prev_layer_name = layer_name
			layer_name = 'conv_{}_{}'.format(Encoder_filters[i], i)
			num_filters = Encoder_filters[i]
			end_points_names.append(layer_name)
			end_points[layer_name] = conv_block(end_points[prev_layer_name],
									num_filters, layer_name, 4,2, self.is_train,
									'batch'  if i else None, activation='leaky')

		end_points_names.pop()

		## Forming the layers in the decoder
		for i in range(len(Decoder_filters)):
			prev_layer_name = layer_name
			layer_name = 'deconv_{}_{}'.format(Decoder_filters[i], i)
			num_filters = Decoder_filters[i]
			end_points[layer_name] = deconv_block(end_points[prev_layer_name],
										num_filters, layer_name, 4,2, self.is_train,
										'batch', activation='relu')
			end_points[layer_name] = tf.concat([end_points[layer_name], 
									end_points[end_points_names.pop()]], axis=3)
		
		## Final layer 
		prev_layer_name = layer_name
		layer_name = 'final'
		end_points[layer_name] = deconv_block(end_points[prev_layer_name], 3,
								layer_name, 4,2, self.is_train, None, activation = 'tanh')

		return end_points['final']


class Discriminator(object):
	""" Discriminator Architecture """
	def __init__(self, image_size, is_train):
		self.image_size = image_size
		self.is_train = is_train
	def __call__(self, input):
		## Number of filters in each of the layers
		disc_filters = [64, 128, 256 ] + [512]*3
		end_points = {}
		layer_name = 'init'
		end_points[layer_name] = input  
		
		## creating the layers
		for i in range(6):
			prev_layer_name = layer_name
			layer_name = 'conv_{}_{}'.format(disc_filters[i],i)
			end_points[layer_name] = conv_block(end_points[prev_layer_name], disc_filters[i], 
											layer_name, 4,2, self.is_train, 
											norm= None if i else 'batch', 
											activation='leaky')
		## creating the final layer
		prev_layer_name = layer_name
		layer_name = 'pre_final'
		end_points[layer_name] = conv_block(end_points[prev_layer_name], 1, 'conv_1_1',
									4,1, self.is_train, norm=None, activation=None,
									bias=True)
		end_points['final'] = tf.reduce_mean(end_points['pre_final'], axis=[1,2,3])
		return end_points['final']

class Encoder(object):
	## 
	def __init__(self, image_size, is_train, latent_dim ):
		self.is_train = is_train
		self.image_size = image_size
		self.latent_dim = latent_dim

	def __call__(self, input):
		enc_filters = [128, 256] + [512]*3
		end_points = {}
		## initial layer creation
		end_points['init'] = input
		layer_name = 'conv_64_0'
		end_points[layer_name] = conv_block(end_points['init'], 64, layer_name,
									4,2, self.is_train, norm=None, activation='leaky', bias=True)
		## creating the res blocks 
		for i in range(len(enc_filters)):
			prev_layer_name = layer_name
			layer_name = 'res_block_{}_{}'.format(enc_filters[i], i+1)

			end_points[layer_name] = residual(end_points[prev_layer_name], enc_filters[i], 
												layer_name, self.is_train, norm='batch',
												bias=True)
			prev_layer_name = layer_name
			layer_name = 'avg_pool_{}'.format(i+1)
			end_points[layer_name] =  tf.nn.avg_pool(end_points[prev_layer_name],
										[1,2,2,1], [1,2,2,1], 'SAME')
		## creating the final layers.
		end_points['relu'] = tf.nn.relu(end_points[layer_name])
		end_points['avg_pool'] = tf.nn.avg_pool(end_points['relu'], [1,8,8,1], [1,8,8,1],
									'SAME')
		end_points['flatten'] = flatten(end_points['avg_pool'])
		mu = fc(end_points['flatten'], self.latent_dim, 'mu', self.is_train, 
					norm=None, activation=None)
		log_sigma = fc(end_points['flatten'], self.latent_dim, 'log_sigma', self.is_train, 
						norm=None, activation=None)
		z = mu + tf.random_normal(shape=tf.shape(self.latent_dim))*tf.exp(log_sigma)
		return z, mu, log_sigma



	