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

class FedAwS:
	def __init__(self):
		self.graph = tf.Graph()
		gpu_options = tf.GPUOptions(allow_growth=True)
		tf_config = tf.ConfigProto(gpu_options=gpu_options,
				allow_soft_placement=True, log_device_placement=False)
		self.sess = tf.Session(graph=self.graph, config=tf_config)
			
	def initialize(self, config, num_classes):
		self.num_classes = num_classes
		'''
			Initialize the graph from scratch according config.
		'''
		with self.sess.as_default():
			with self.graph.as_default():
			
				# Set up placeholders
				w, h = config.image_size
				channels = config.channels

				# embeddings = tf.placeholder(tf.float32, shape=[self.num_classes,512], name='class_embeddings')

				learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
				keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob')
				phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

				# normalized_embeddings = tf.nn.l2_normalize(embeddings, dim=1, name='normalized_embeddings')
				global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

				num_classes = 1000
				num_features = 512

				weights = tf.get_variable('trainable_class_embeddings', shape=(num_classes, num_features),
						# initializer=slim.xavier_initializer(),
						initializer=tf.constant_initializer(0),
						trainable=True,
						dtype=tf.float32)
				self.weights = tf.nn.l2_normalize(weights, dim=1)

				tf.add_to_collection('class_embeddings', self.weights)


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
								
								# Build all losses
								losses = []

								spreadout_loss = tflosses.spreadout_regularizer(self.weights, global_step, margin=-0.3, reduce_mean=True, scope='spreadout_regularizer', reuse=None)
								losses.append(spreadout_loss)
								insert_dict('spreadout_loss', spreadout_loss)

							   # Collect all losses
								reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
								losses.append(reg_loss)
								insert_dict('reg_loss', reg_loss)

								total_loss = tf.add_n(losses, name='total_loss')
								grads_split = tf.gradients(total_loss, [var for var in tf.trainable_variables()])
								grads_splits.append(grads_split)

								# Keep useful tensors
								# if i == 0:
								# 	self.inputs = images
								# 	self.outputs = embeddings
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
				apply_gradient_op = tfutils.apply_gradient([var for var in tf.trainable_variables()], grads, config.optimizer,
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
				self.learning_rate_placeholder = learning_rate_placeholder 
				self.keep_prob_placeholder = keep_prob_placeholder 
				self.phase_train_placeholder = phase_train_placeholder
				self.global_step = global_step
				self.train_op = train_op
				self.summary_op = summary_op
	
	def get_trainable_variables(self,):
		with self.graph.as_default():
			with self.sess.as_default():
				print(tf.trainable_variables())
		return 

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

	def set_model_weights(self, class_embeddings):
		with self.graph.as_default():
			with self.sess.as_default():
				trainable_variables = tf.trainable_variables()
				for variable in trainable_variables:
					if('trainable_class_embeddings' in variable.name):
						self.sess.run(variable.assign(class_embeddings))
		return


	def train(self, learning_rate, keep_prob):
		with self.graph.as_default():
			with self.sess.as_default():
				feed_dict = {self.learning_rate_placeholder: learning_rate,
							self.keep_prob_placeholder: keep_prob,
							self.phase_train_placeholder: True}

				_, wl, sm = self.sess.run([self.train_op, tfwatcher.get_watchlist(), self.summary_op], feed_dict = feed_dict)
				step = self.sess.run(self.global_step)

		return wl, sm, step

	def get_mean_spreadout(self,):
		feed_dict = {self.phase_train_placeholder: False,
						self.keep_prob_placeholder: 1.0}
		weights = self.sess.run(self.weights, feed_dict=feed_dict)
		similarity = np.matmul(weights, np.transpose(weights))
		mask_non_diag = np.logical_not(np.eye(self.num_classes))

		similarity = similarity[np.where(mask_non_diag)]
		print(np.mean(similarity))

	def get_class_embeddings(self,):
		feed_dict = {self.phase_train_placeholder: False,
						self.keep_prob_placeholder: 1.0}
		weights = self.sess.run(self.weights, feed_dict=feed_dict)
		return weights



	
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

		
