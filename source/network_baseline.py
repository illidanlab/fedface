"""Main training file for face recognition
"""
# MIT License
# 
# Copyright (c) 2017 Yichun Shi
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import imp
import time

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


import nntools.tensorflow as tftools
import nntools.tensorflow.utils as tfutils
import nntools.tensorflow.losses as tflosses
import nntools.tensorflow.watcher as tfwatcher

class Network:
	def __init__(self):
		self.graph = tf.Graph()
		gpu_options = tf.GPUOptions(allow_growth=True)
		tf_config = tf.ConfigProto(gpu_options=gpu_options,
				allow_soft_placement=True, log_device_placement=False)
		self.sess = tf.Session(graph=self.graph, config=tf_config)
			
	def initialize(self, config, num_classes):
		'''
			Initialize the graph from scratch according config.
		'''
		with self.graph.as_default():
			with self.sess.as_default():
				# Set up placeholders
				w, h = config.image_size
				channels = config.channels
				image_batch_placeholder = tf.placeholder(tf.float32, shape=[None, h, w, channels], name='image_batch')
				label_batch_placeholder = tf.placeholder(tf.int32, shape=[None], name='label_batch')
				weight_batch_placeholder = tf.placeholder(tf.float32, shape=[None], name='weight_batch')
				learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
				keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob')
				phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
				switch_dist_placeholder = tf.placeholder(tf.float32, name='switch_dist')
				switch_sigma_placeholder = tf.placeholder(tf.float32, name='switch_sigma')
				global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

				image_splits = tf.split(image_batch_placeholder, config.num_gpus)
				label_splits = tf.split(label_batch_placeholder, config.num_gpus)
				weight_splits = tf.split(weight_batch_placeholder, config.num_gpus)
				grads_splits = []
				split_dict = {}
				def insert_dict(k,v):
					if k in split_dict: split_dict[k].append(v)
					else: split_dict[k] = [v]
						
				for i in range(config.num_gpus):
					scope_name = '' if i==0 else 'gpu_%d' % i
					with tf.name_scope(scope_name):
						with tf.variable_scope('', reuse=i>0):
							with tf.device('/gpu:%d' % i):
								images = tf.identity(image_splits[i], name='inputs')
								labels = tf.identity(label_splits[i], name='labels')
								sample_weights = tf.identity(weight_splits[i], name='sample_weights')
								
								# Build networks

								network = imp.load_source('network_define', config.network)
								prelogits = network.inference(images, keep_prob_placeholder, phase_train_placeholder,
														bottleneck_layer_size = config.embedding_size, 
														weight_decay = config.weight_decay)
								prelogits = tf.identity(prelogits, name='prelogits')
								embeddings = tf.nn.l2_normalize(prelogits, dim=1, name='embeddings')
								# embeddings = tflosses.group_normalize(prelogits, 4, name='embeddings')
								# embeddings = tf.identity(prelogits, name='embeddings')
								# sigma_sq = tf.identity(tf.exp(log_sigma_sq), name='sigma_sq')
								# sigma_sq = tf.identity(tf.nn.sigmoid(log_sigma_sq) / 512, name='sigma_sq')
								# sigma_sq = tf.identity(1e-6 + tf.nn.softplus(log_sigma_sq, name='sigma_sq'))

								# Build all losses
								losses = []

								# samples = prelogits + tf.exp(0.5*log_sigma_sq) * tf.random_normal(tf.shape(prelogits))

								if False:
									input_decoder = tf.cond(phase_train_placeholder, lambda: prelogits, lambda: samples)
									input_decoder = tf.identity(prelogits, name='input_decoder')
									rec = network.decoder(input_decoder, keep_prob_placeholder, phase_train_placeholder,
												weight_decay = config.weight_decay, model_version = config.model_version)
									rec = tf.identity(rec, name='output_decoder')

									if i == 0:
										rec_ = network.decoder(samples, keep_prob_placeholder, phase_train_placeholder,
											weight_decay = config.weight_decay, model_version = config.model_version, reuse=True)
										image_grid = tf.stack([images, rec, rec_], axis=1)[:12]
										image_grid = tf.reshape(image_grid, [-1, h, w, channels])
										image_grid = tftools.image_ops.image_grid(image_grid, (3,12))
										tf.summary.image('image_grid', image_grid)

								# tfwatcher.insert('mean_norm', tf.reduce_mean(tf.norm(samples, axis=1)))
								# tfwatcher.insert('mean_sigma', tf.reduce_mean(tf.exp(0.5*log_sigma_sq)))


								# Orignal Softmax
								if 'softmax' in config.losses.keys():
									softmax_loss = tflosses.softmax_loss(samples, labels, num_classes,
													weight_decay=config.weight_decay, **config.losses['softmax'])
									losses.append(softmax_loss)
									insert_dict('sfloss', softmax_loss)
								# Triplet Loss
								if 'triplet' in config.losses.keys():
									triplet_loss = tflosses.triplet_semihard_loss(labels, prelogits, **config.losses['triplet'])
									losses.append(triplet_loss)
									insert_dict('loss', triplet_loss)
								# Uncertain Triplet Loss
								if 'utriplet' in config.losses.keys():
									uncertain_triplet_loss = tflosses.uncertain_triplet_loss(labels, prelogits, log_sigma_sq, **config.losses['utriplet'])
									losses.append(uncertain_triplet_loss)
									insert_dict('uloss', uncertain_triplet_loss)
								# Uncertain Pair Loss
								if 'upair' in config.losses.keys():
									uncertain_pair_loss = tflosses.uncertain_pair_loss(prelogits, log_sigma_sq, **config.losses['upair'])
									losses.append(uncertain_pair_loss)
									insert_dict('uploss', uncertain_pair_loss)
								# L2-Softmax
								if 'cosine' in config.losses.keys():
									logits, cosine_loss = tflosses.cosine_softmax(prelogits, labels, num_classes, 
															weight_decay=config.weight_decay, **config.losses['cosine'])
									losses.append(cosine_loss)
									insert_dict('closs', cosine_loss)
								# A-Softmax
								if 'angular' in config.losses.keys():
									a_cfg = config.losses['angular']
									angular_loss = tflosses.angular_softmax(prelogits, labels, num_classes, 
															weight_decay=config.weight_decay, **config.losses['angular'])
									losses.append(angular_loss)
									insert_dict('aloss', angular_loss)
								# Split Loss
								if 'split' in config.losses.keys():
									split_loss = tflosses.split_softmax(prelogits, labels, num_classes, 
															global_step, weight_decay=config.weight_decay,
															**config.losses['split'])
									losses.append(split_loss)
									insert_dict('loss', split_loss)
								# AM-Softmax
								if 'am_softmax' in config.losses.keys():
									amloss = tflosses.am_softmax(prelogits, labels, num_classes, 
															global_step, weight_decay=config.weight_decay,
															**config.losses['am_softmax'])
									losses.append(amloss)
									insert_dict('loss', amloss)
								# AM-Softmax
								if 'group' in config.losses.keys():
									group_weights = log_sigma_sq
									gloss = tflosses.group_loss(prelogits, group_weights, labels, num_classes, 
															global_step, weight_decay=config.weight_decay,
															**config.losses['group'])
									losses.append(gloss)
									insert_dict('gloss', gloss)
								# Conditional Loss
								if 'conditional' in config.losses.keys():
									conditional_loss = tflosses.conditional_loss(prelogits, log_sigma_sq, labels, num_classes, 
															global_step, **config.losses['conditional'])
									losses.append(conditional_loss)
									insert_dict('loss', conditional_loss)
								# Stochastic Loss
								if 'stochastic' in config.losses.keys():
									stochastic_loss = tflosses.stochastic_loss(prelogits, log_sigma_sq, labels, num_classes, 
															global_step, weight_decay=config.weight_decay,
															**config.losses['stochastic'])
									losses.append(stochastic_loss)
									insert_dict('stloss', stochastic_loss)
								# Divergence Loss
								if 'divergence' in config.losses.keys():
									divergence_loss = tflosses.class_divergence(prelogits, log_sigma_sq, labels, num_classes, 
															global_step, weight_decay=config.weight_decay,
															**config.losses['divergence'])
									losses.append(divergence_loss)
									insert_dict('divloss', divergence_loss)
								# KL Divergence with Standard Normal
								if 'kl_std' in config.losses.keys():
									kl_loss = tflosses.gaussian_kl_divergence(prelogits, log_sigma_sq, 0.0, 0.0)
									kl_loss = config.losses['kl_std']['coef'] * tf.reduce_mean(kl_loss)
									losses.append(kl_loss)
									insert_dict('klloss', kl_loss)
								# Dimension Pooling Loss
								if 'dploss' in config.losses.keys():
									dp_loss = tflosses.dim_pool(prelogits, -log_sigma_sq, labels, num_classes,
										global_step, config.weight_decay, learning_rate_placeholder, **config.losses['dploss'])
									dp_loss = tf.identity(dp_loss, name='dploss')
									losses.append(dp_loss)
									insert_dict('dploss', dp_loss)
								# Norm Loss
								if 'norm' in config.losses.keys():
									norm_loss = tflosses.norm_loss(prelogits, **config.losses['norm'])
									losses.append(norm_loss)
									insert_dict('loss', norm_loss)
								# Rec Loss
								if 'rec' in config.losses.keys():
									rec_loss = config.losses['rec']['coef'] * \
										tf.reduce_mean(tf.abs(images - rec), name='rec_loss')
									losses.append(rec_loss)
									insert_dict('rloss', rec_loss)

								if 'squared_hinge_loss' in config.losses.keys():
									sqaured_hinge_loss = tflosses.squared_hinge_loss_with_cosine_distance(prelogits, class_embedding, 
										global_step, **config.losses['squared_hinge_loss'] )

							   # Collect all losses
								reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
								losses.append(reg_loss)
								insert_dict('reg_loss', reg_loss)

								total_loss = tf.add_n(losses, name='total_loss')
								grads_split = tf.gradients(total_loss, tf.trainable_variables())
								grads_splits.append(grads_split)

								# Keep useful tensors
								if i == 0:
									self.inputs = images
									self.outputs = prelogits
									# self.sigma_sq = sigma_sq


				# Merge the splits
				grads = tfutils.average_grads(grads_splits)
				for k,v in split_dict.items():
					v = tfutils.average_tensors(v)
					tfwatcher.insert(k, v)
					if 'loss' in k:
						tf.summary.scalar('losses/' + k, v)
					else:
						tf.summary.scalar(k, v)


				# Training Operaters
				apply_gradient_op = tfutils.apply_gradient(tf.trainable_variables(), grads, config.optimizer,
										learning_rate_placeholder, config.learning_rate_multipliers)

				update_global_step_op = tf.assign_add(global_step, 1)

				update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

				# train_ops = [apply_gradient_op, update_global_step_op] + update_ops
				train_op = tf.group(apply_gradient_op, update_global_step_op, *update_ops)

				tf.summary.scalar('learning_rate', learning_rate_placeholder)
				summary_op = tf.summary.merge_all()

				# Initialize variables
				self.sess.run(tf.local_variables_initializer())
				self.sess.run(tf.global_variables_initializer())
				self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=None)

				# Keep useful tensors
				self.config = config
				self.image_batch_placeholder = image_batch_placeholder
				self.label_batch_placeholder = label_batch_placeholder
				self.weight_batch_placeholder = weight_batch_placeholder
				self.learning_rate_placeholder = learning_rate_placeholder 
				self.keep_prob_placeholder = keep_prob_placeholder 
				self.phase_train_placeholder = phase_train_placeholder
				self.switch_dist_placeholder = switch_dist_placeholder
				self.switch_sigma_placeholder = switch_sigma_placeholder
				self.global_step = global_step
				self.train_op = train_op
				self.summary_op = summary_op
				
	def get_model_weights(self):
		import pdb
		weights = []
		with self.graph.as_default():
			with self.sess.as_default():
				trainable_variables = tf.trainable_variables()
				for variable in trainable_variables:
					if('AM-Softmax' in variable.name):
						continue
					weights.append(self.sess.run(variable))
		return weights

	def set_model_weights(self, weights):
		with self.graph.as_default():
			with self.sess.as_default():
				trainable_variables = tf.trainable_variables()
				i = 0
				for variable in trainable_variables:
					if('AM-Softmax' in variable.name):
						continue
					self.sess.run(variable.assign(weights[i]))
					i = i + 1
		return


	def train(self, image_batch, label_batch, learning_rate, keep_prob):
		feed_dict = {self.image_batch_placeholder: image_batch,
					self.label_batch_placeholder: label_batch,
					self.learning_rate_placeholder: learning_rate,
					self.keep_prob_placeholder: keep_prob,
					self.phase_train_placeholder: True}

		_, wl, sm = self.sess.run([self.train_op, tfwatcher.get_watchlist(), self.summary_op], feed_dict = feed_dict)
		step = self.sess.run(self.global_step)

		return wl, sm, step
	
	def restore_model(self, *args, **kwargs):
		trainable_variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		trainable_variables = [var for var in trainable_variables if 'SphereNet/BatchNorm/' not in var.name]
		tfutils.restore_model(self.sess, trainable_variables, *args, **kwargs)

	def update_class_stats(self, new_mu, new_log_sigma_sq):
		# Additional Operators
		with self.graph.as_default():
			if not hasattr(self, 'update_class_mu_op'):
				class_mu = tf.get_collection('class_mu')[0]
				print('class_mu tensor: {}'.format(class_mu))
				self.class_mu_placeholder = tf.placeholder(tf.float32, shape=class_mu.shape, name='class_mu')
				self.update_class_mu_op = tf.assign(class_mu, self.class_mu_placeholder)

				class_log_sigma_sq = tf.get_collection('class_log_sigma_sq')[0]
				print('class_log_sigma_sq tensor: {}'.format(class_log_sigma_sq))
				self.class_log_sigma_sq_placeholder = tf.placeholder(tf.float32, shape=class_log_sigma_sq.shape, name='class_log_sigma_sq')
				self.update_class_log_sigma_sq_op = tf.assign(class_log_sigma_sq, self.class_log_sigma_sq_placeholder)

			feed_dict = {self.class_mu_placeholder: new_mu, self.class_log_sigma_sq_placeholder: new_log_sigma_sq}
			_ = self.sess.run([self.update_class_mu_op, self.update_class_log_sigma_sq_op], feed_dict = feed_dict)
		

	def save_model(self, model_dir, global_step):
		tfutils.save_model(self.sess, self.saver, model_dir, global_step)
		

	def load_model(self, model_path, scope=None):
		tfutils.load_model(self.sess, model_path, scope=scope)
		self.phase_train_placeholder = self.graph.get_tensor_by_name('phase_train:0')
		self.keep_prob_placeholder = self.graph.get_tensor_by_name('keep_prob:0')
		self.inputs = self.graph.get_tensor_by_name('image_batch:0')
		self.outputs = self.graph.get_tensor_by_name('prelogits:0')
		# self.sigma_sq = self.graph.get_tensor_by_name('sigma_sq:0')
		try:
			self.input_decoder = self.graph.get_tensor_by_name('input_decoder:0')
			self.output_decoder = self.graph.get_tensor_by_name('output_decoder:0')
		except:
			print(':: Decoder is NOT found in the graph.')
		self.config = imp.load_source('network_config', os.path.join(model_path, 'config.py'))

	def reconstruct_faces(self, latent, batch_size):
		num_images = len(latent)
		result = np.ndarray([num_images]+list(self.output_decoder.shape[1:]), dtype=np.float32)
		start_time = time.time()
		for start_idx in range(0, num_images, batch_size):
			end_idx = min(num_images, start_idx + batch_size)
			inputs = latent[start_idx:end_idx]
			feed_dict = {self.input_decoder: inputs,
						self.phase_train_placeholder: False,
						self.keep_prob_placeholder: 1.0}
			result[start_idx:end_idx] = self.sess.run(self.output_decoder, feed_dict=feed_dict)
		return result

	def sample_faces(self, images, batch_size, proc_func=None, verbose=False):
		num_images = len(images)
		result = np.ndarray(images.shape, dtype=np.float32)
		start_time = time.time()
		for start_idx in range(0, num_images, batch_size):
			if verbose:
				elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))
				sys.stdout.write('# of images: %d Current image: %d Elapsed time: %s \t\r' 
					% (num_images, start_idx, elapsed_time))
			end_idx = min(num_images, start_idx + batch_size)
			inputs = images[start_idx:end_idx]
			if proc_func:
				inputs = proc_func(inputs)
			feed_dict = {self.inputs: inputs,
						self.phase_train_placeholder: False,
						self.keep_prob_placeholder: 1.0}
			result[start_idx:end_idx]  = self.sess.run(self.output_decoder, feed_dict=feed_dict)
		if verbose:
			print('')
		return result

	def extract_feature(self, images, batch_size, proc_func=None, verbose=False):
		num_images = len(images)
		num_features = self.outputs.shape[1]
		# num_weights = self.sigma_sq.shape[1]
		result = np.ndarray((num_images, num_features), dtype=np.float32)
		# sigma_sq = np.ndarray((num_images, num_weights), dtype=np.float32)
		start_time = time.time()
		for start_idx in range(0, num_images, batch_size):
			if verbose:
				elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))
				sys.stdout.write('# of images: %d Current image: %d Elapsed time: %s \t\r' 
					% (num_images, start_idx, elapsed_time))
			end_idx = min(num_images, start_idx + batch_size)
			inputs = images[start_idx:end_idx]
			if proc_func:
				inputs = proc_func(inputs)
			feed_dict = {self.inputs: inputs,
						self.phase_train_placeholder: False,
					self.keep_prob_placeholder: 1.0}
			result[start_idx:end_idx]= self.sess.run(self.outputs, feed_dict=feed_dict)
		if verbose:
			print('')
		return result

		
