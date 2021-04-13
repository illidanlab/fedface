import os
import sys
import time
import argparse
import tensorflow as tf
import numpy as np
from functools import partial
import random
import utils
# import visualize
import facepy
from nntools.common.dataset import Dataset, single_subject_per_client

from nntools.common.imageprocessing import preprocess, flip

from network_baseline import Network
from network_federated import Network as Network_Federated
from FedAwS import FedAwS
from sklearn.model_selection import KFold
import math
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import json
import pickle
import pdb

import shutil
import matplotlib as mpl
# from scipy import misc
import csv
import scipy.misc
from lfw import LFWTest
from ijba import IJBATest
from ijbc import IJBCTest

import multiprocessing
# multiprocessing.get_context('spawn')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import psutil

def test_LFW(lfw, features=None, matcher=None, model=None):
	if(model is None):
		if(features is None):
			results_lfw = lfw.test_standard_proto(facepy.metric.cosine_pair, features=None, matcher=matcher, batch_size_preprocess=10000, batch_size_extract=1024)
		else:
			features = facepy.linalg.normalize(features)
			results_lfw = lfw.test_standard_proto(facepy.metric.cosine_pair, features=features, matcher=None, batch_size_preprocess=10000, batch_size_extract=1024)
	else:
		features = model.generate_cosface_feats(lfw.image_paths, lfw.config, batch_size=128)
		results_lfw = lfw.test_standard_proto(facepy.metric.cosine_pair, features=features, matcher=None, batch_size_preprocess=10000, batch_size_extract=1024)
	return results_lfw

def weight_scaling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    #get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    #first calculate the total training data points across clients
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    return local_count/global_count


def weight_scaling_factor(trainset, client_id, chosen_clients):
	global_count = sum([client.images.shape[0] for client in trainset[chosen_clients]])
	local_count = trainset[client_id].images.shape[0]
	return local_count/global_count

def scale_model_weights(weight, scalar):
		'''function for scaling a models weights'''
		weight_final = []
		steps = len(weight)
		for i in range(steps):
			weight_final.append(scalar * weight[i])
		return weight_final

def sum_scaled_weights(scaled_weight_list):
	'''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
	avg_grad = list()
	#get the average grad accross all client gradients
	for grad_list_tuple in zip(*scaled_weight_list):
		layer_mean = np.sum(grad_list_tuple, axis=0)
		avg_grad.append(layer_mean)
		
	return avg_grad

def train_client(global_weights, trainset, client, config, proc_func_train, proc_func_test, ijba, log_dir):
	trainset[client].start_batch_queue(config.local_batch_size, batch_format=config.batch_format, proc_func=proc_func_train)
	local_step = 0
	network_local = Network()
	network_local.initialize(config, trainset[client].num_classes)
	network_local.set_model_weights(global_weights)

	start_time = time.time()

	for epoch in range(config.num_local_epochs):
		for step in range(config.local_epoch_size):
			learning_rate = utils.get_updated_learning_rate(local_step, config)
			batch = trainset[client].pop_batch_queue()
			wl, sm, local_step = network_local.train(batch['images'], batch['labels'], learning_rate, config.keep_prob)
			duration = time.time() - start_time
			start_time = time.time()
			utils.display_info(epoch, step, duration, wl)
			wl['lr'] = learning_rate
		ijba.extract_features(proc_func_test, matcher=network_local)
		sr_test_local = ijba.test_verification(facepy.metric.cosine_pair)
		with open(log_dir + '/accuracy_' + str(client)+'.txt', 'a') as f:
			f.write('{}: {}\n'.format(local_step, sr_test_local))

	return network_local.get_model_weights()

def train_client_positive_labels(network_local, trainset, client, config, proc_func_train, proc_func_test, ijba, log_dir, class_embedding):
	

	local_step = 0

	start_time = time.time()
	for epoch in range(config.num_local_epochs):
		for step in range(config.local_epoch_size):
			learning_rate = utils.get_updated_learning_rate(local_step, config)
			batch = trainset[client].pop_batch_queue()
			wl, sm, local_step = network_local.train(batch['images'], batch['labels'], class_embedding, learning_rate, config.keep_prob)
			duration = time.time() - start_time
			start_time = time.time()
			utils.display_info(epoch, step, duration, wl)
			wl['lr'] = learning_rate
		# ijba.extract_features(proc_func_test, matcher=network_local)
		# sr_test_local = ijba.test_verification(facepy.metric.cosine_pair)
		# with open(log_dir + '/accuracy_' + str(client)+'.txt', 'a') as f:
		# 	f.write('{}: {}\n'.format(local_step, sr_test_local))
	model_weights = network_local.get_model_weights()
	return model_weights, wl

def get_mean_class_embeddings(trainset, network, proc_func_train):
	mean_class_embeddings = []

	for subject in trainset:
		temp_image_paths = subject.images
		temp_imgs = proc_func_train(temp_image_paths)
		temp_features = network.extract_feature(temp_imgs, 512)
		mean_class_embeddings.append(np.mean(temp_features, axis=0))
	return np.array(mean_class_embeddings)

def train_federated_one_subject_per_client(config, comm_round, class_embeddings,log_dir,global_weights=None):


	total_num_clients = 1000
	num_clients = 2

	trainset = single_subject_per_client('/home/divyansh/Research/Federated_Learning/train_federated/datasets/federated_train.txt', prefix='')
	network_global = Network_Federated()
	network_global.initialize(config, trainset[0].num_classes)

	if(global_weights == None):
		network_global.load_model('/home/divyansh/Research/Federated_Learning/train_federated/log/CASIA_9000_subjects_baseline/20210206-092109')
		global_weights = network_global.get_model_weights()
	# else:
	network_global = Network_Federated()
	network_global.initialize(config, trainset[0].num_classes)
	network_global.set_model_weights(global_weights)

	# print(network_global.get_trainable_variables())
	# pdb.set_trace()

	ijba = IJBATest('/media/divyansh/data/Datasets/IJB-A/list_ijba_retina_aligned_112.txt')
	ijba.init_verification_proto('/media/divyansh/data/Datasets/IJB-A/IJB-A_11_sets/')
	proc_func_test = lambda images: preprocess(images, config, False)

	# print('Printing Pretrained Accuracies\n\n\n')
	# ijba.extract_features(proc_func_test, matcher=network_global)
	# sr_test = ijba.test_verification(facepy.metric.cosine_pair)
	# print('IJBA Verification Accuracy ', sr_test)

	# lfw = LFWTest('/media/divyansh/data/Datasets/LFW/lfw_imnames.txt', config)
	# lfw.init_standard_proto('/media/divyansh/data/Datasets/LFW/pairs.txt')
	# acc_lfw_original = test_LFW(lfw, features=None, matcher=network_global)

	summary_writer = tf.summary.FileWriter(log_dir, network_global.graph)
		
	proc_func_train = lambda images: preprocess(images, config, True)

	print('\nStart Training\n# epochs: %d\nepoch_size: %d\nbatch_size: %d\n'\
		% (config.num_comm_rounds, config.num_local_epochs, config.local_batch_size))
	global_step = 0
	start_time = time.time()
	BEST = 0.0
	temp_step = 0


	network_local = Network_Federated()
	network_local.initialize(config, trainset[0].num_classes)
	network_local.set_model_weights(global_weights)

	# chosen_clients = np.random.choice(range(0,1000), num_clients)

	scaled_local_weights_list = []
	for i,client in enumerate(range(num_clients)):
		print('Client : ', i, client)
		trainset[client].start_batch_queue(config.local_batch_size, batch_format=config.batch_format, proc_func=proc_func_train)
		scaled_weights, wl = train_client_positive_labels(network_local, trainset, client, config, proc_func_train, proc_func_test, ijba, log_dir, class_embeddings[client])
		trainset[client].release_queue()
		scaled_weights = scale_model_weights(scaled_weights, weight_scaling_factor(trainset, client, range(num_clients)))



		if(len(scaled_local_weights_list) == 0):
			scaled_local_weights_list.append(scaled_weights)
		else:
			# pdb.set_trace()
			scaled_local_weights_list.append(scaled_weights)
			average_weights = sum_scaled_weights(scaled_local_weights_list)
			scaled_local_weights_list = []
			scaled_local_weights_list.append(average_weights)
		del scaled_weights
		# ijba.extract_features(proc_func_test, matcher=network_local)
		# sr_test = ijba.test_verification(facepy.metric.cosine_pair)
		# print(sr_test)
		# network_local.set_model_weights(global_weights)


	# pdb.set_trace()
	network_global.set_model_weights(scaled_local_weights_list[0])
	duration = time.time() - start_time
	start_time = time.time()
	utils.display_info(comm_round, 0, duration, wl)
	network_global.save_model(log_dir, comm_round)

	# ijba.extract_features(proc_func_test, matcher=network_global)
	# sr_test = ijba.test_verification(facepy.metric.cosine_pair)
	# print(sr_test)

	# with open(log_dir + '/accuracy.txt', 'a') as f:
	# 	f.write('{}: {}\n'.format(temp_step, sr_test))

	temp_step = temp_step + 1
		
	summary = tf.Summary()
	# # summary.value.add(tag='fgnet/rank1', simple_value=sr_test)
	summary_writer.add_summary(summary, global_step)
	global_weights = network_global.get_model_weights()
	np.save('./global_weights.npy', np.array(global_weights, dtype=object),allow_pickle=True)

	fedaws = FedAwS()
	fedaws.initialize(config, total_num_clients)
	fedaws.set_model_weights(np.squeeze(class_embeddings))
	fedaws.get_mean_spreadout()

	local_step = 0
	start_time = time.time()
	for epoch in range(10):
		for step in range(config.local_epoch_size):
			learning_rate = utils.get_updated_learning_rate(local_step, config)
			wl, sm, local_step = fedaws.train(learning_rate, config.keep_prob)
			duration = time.time() - start_time
			start_time = time.time()
			utils.display_info(epoch, step, duration, wl)
			wl['lr'] = learning_rate

	fedaws.get_mean_spreadout()
	updated_class_embeddings = fedaws.get_mean_class_embeddings()

	pdb.set_trace()



def train_federated(config, config_file):

	trainset = []
	num_clients = 64

	for client in range(num_clients):
		trainset.append(Dataset('../datasets/CASIA_clients_homogenous_64/client' + str(client) + '.txt', prefix=''))

	trainset = np.array(trainset)

	network_global = Network()
	network_global.initialize(config, trainset[0].num_classes)
	global_weights = network_global.get_model_weights()
	# # network_global.load_model('/home/divyansh/Research/Federated_Learning/train_federated/CASIA_Fed4_baseline_heterogenous_4/20201231-142606')
	# # global_weights = network_global.get_model_weights()
	# network_global = Network()
	# network_global.initialize(config, trainset[0].num_classes)

	ijba = IJBATest('/media/divyansh/data/Datasets/IJB-A/list_ijba_retina_aligned_112.txt')
	ijba.init_verification_proto('/media/divyansh/data/Datasets/IJB-A/IJB-A_11_sets/')
	proc_func_test = lambda images: preprocess(images, config, False)

	# lfw = LFWTest('/media/divyansh/data/Datasets/LFW/lfw_imnames.txt', config)
	# lfw.init_standard_proto('/media/divyansh/data/Datasets/LFW/pairs.txt')
	# acc_lfw_original = test_LFW(lfw, features=None, matcher=network_global)

	# Initalization for running
	
	summary_writer = tf.summary.FileWriter(log_dir, network_global.graph)
		
	proc_func_train = lambda images: preprocess(images, config, True)

	#
	# Main Loop
	#
	print('\nStart Training\n# epochs: %d\nepoch_size: %d\nbatch_size: %d\n'\
		% (config.num_epochs, config.epoch_size, config.batch_size))
	global_step = 0
	start_time = time.time()
	BEST = 0.0
	temp_step = 0

	for comm_round in range(config.num_comm_rounds):
		scaled_local_weights_list = []
		for client in range(num_clients):
			scaled_weights = scale_model_weights(train_client(global_weights, trainset, client, config, proc_func_train, proc_func_test, ijba, log_dir), weight_scaling_factor(trainset, client))
			scaled_local_weights_list.append(scaled_weights)

		average_weights = sum_scaled_weights(scaled_local_weights_list)
		network_global.set_model_weights(average_weights)
		duration = time.time() - start_time
		start_time = time.time()
		utils.display_info(comm_round, 0, duration, wl)
		network_global.save_model(log_dir, temp_step)
		ijba.extract_features(proc_func_test, matcher=network_global)
		sr_test = ijba.test_verification(facepy.metric.cosine_pair)
		print(sr_test)

		with open(log_dir + '/accuracy.txt', 'a') as f:
			f.write('{}: {}\n'.format(temp_step, sr_test))

		temp_step = temp_step + 1
			
		summary = tf.Summary()
		# # summary.value.add(tag='fgnet/rank1', simple_value=sr_test)
		summary_writer.add_summary(summary, global_step)
		global_weights = network_global.get_model_weights()


def train_baseline(config, config_file, dataset_path, prefix):

	trainset = Dataset(dataset_path, prefix=prefix)
	network_global = Network()
	network_global.initialize(config, 9000)

	network_global.load_model('/home/divyansh/Research/Federated_Learning/train_federated/log/Baseline_64_layer/20210405-155445/')
	global_weights = network_global.get_model_weights()

	network_global = Network()
	network_global.initialize(config, trainset.num_classes)
	network_global.set_model_weights(global_weights)

	lfw = LFWTest('/media/divyansh/data/Datasets/LFW/lfw_imnames.txt', config)
	lfw.init_standard_proto('/media/divyansh/data/Datasets/LFW/pairs.txt')
	acc_lfw_original = test_LFW(lfw, features=None, matcher=network_global)

	ijba = IJBATest('/media/divyansh/data/Datasets/IJB-A/list_ijba_retina_aligned_112.txt')
	ijba.init_verification_proto('/media/divyansh/data/Datasets/IJB-A/IJB-A_11_sets/')
	proc_func_test = lambda images: preprocess(images, config, False)

	ijba.extract_features(proc_func_test, matcher=network_global)
	sr_test_ijba = ijba.test_verification(facepy.metric.cosine_pair)
	print('IJBA Verification Accuracy : ', sr_test_ijba)

	# Initalization for running
	log_dir = utils.create_log_dir(config, config_file)
	os.mkdir(os.path.join(log_dir, 'samples'))
	summary_writer = tf.summary.FileWriter(log_dir, network_global.graph)
		
	proc_func_train = lambda images: preprocess(images, config, True)
	trainset.start_batch_queue(config.batch_size, batch_format=config.batch_format, proc_func=proc_func_train)

	#
	# Main Loop
	#
	print('\nStart Training\n# epochs: %d\nepoch_size: %d\nbatch_size: %d\n'\
		% (config.num_epochs, config.epoch_size, config.batch_size))
	global_step = 0
	start_time = time.time()
	BEST = 0.0

	for epoch in range(config.num_epochs):
		# Training
		for step in range(config.epoch_size):
			# Prepare input
			learning_rate = utils.get_updated_learning_rate(global_step, config)
			batch = trainset.pop_batch_queue()

			wl, sm, global_step = network_global.train(
				batch['images'], batch['labels'],
				learning_rate, config.keep_prob)

			wl['lr'] = learning_rate

			# Display
			if step % config.summary_interval == 0:
				duration = time.time() - start_time
				start_time = time.time()
				utils.display_info(epoch, step, duration, wl)
				summary_writer.add_summary(sm, global_step=global_step)

		network_global.save_model(log_dir, global_step)

		acc_lfw_original = test_LFW(lfw, features=None, matcher=network_global)

		ijba.extract_features(proc_func_test, matcher=network_global)
		sr_test_ijba = ijba.test_verification(facepy.metric.cosine_pair)
		print('IJBA Verification Accuracy : ', sr_test_ijba)

		ijbc.extract_features(proc_func_test, matcher=network_global)
		sr_test_ijbc = ijbc.test_verification(facepy.metric.cosineSimilarity)
		print('IJBC Verification Accuracy : ', sr_test_ijbc)

		with open(log_dir + '/accuracy.txt', 'a') as f:
			f.write('{}: {} ; {}\n'.format(global_step, sr_test_ijba, acc_lfw_original, sr_test_ijbc))

		summary = tf.Summary()
		# summary.value.add(tag='fgnet/rank1', simple_value=sr_test)
		summary_writer.add_summary(summary, global_step)

def evaluate(config, model_path):

	network = Network()
	# network.initialize(config, trainset[0].num_classes)
	network.load_model(model_path)

	lfw = LFWTest('/media/divyansh/data/Datasets/LFW/lfw_imnames.txt', config)
	lfw.init_standard_proto('/media/divyansh/data/Datasets/LFW/pairs.txt')
	acc_lfw_original = test_LFW(lfw, features=None, matcher=network)

	ijba = IJBATest('/media/divyansh/data/Datasets/IJB-A/list_ijba_retina_aligned_112.txt')
	ijba.init_verification_proto('/media/divyansh/data/Datasets/IJB-A/IJB-A_11_sets/')
	proc_func_test = lambda images: preprocess(images, config, False)

	ijba.extract_features(proc_func_test, matcher=network)
	sr_test = ijba.test_verification(facepy.metric.cosine_pair)
	print('IJBA Verification Accuracy : ', sr_test)



def main(args):
	config_file = args.config_file
	# I/O
	config = utils.import_file(config_file, 'config')

	# global global_weights

	# class_embeddings = np.random.normal(loc=0.0, scale=1.0, size=(1000, 512, 1))
	# class_embeddings = get_mean_class_embeddings(trainset, network_global, proc_func_train)
	# class_embeddings = np.load('mean_class_embeddings.npy', allow_pickle=True)
	# class_embeddings = np.expand_dims(class_embeddings, axis=-1)

	# log_dir = utils.create_log_dir(config, config_file)
	# os.mkdir(os.path.join(log_dir, 'samples'))

	train_baseline(config,config_file, '/home/divyansh/Research/Federated_Learning/train_federated/datasets/9000_train.txt', prefix='')
	import sys
	sys.exit()
	# evaluate(config, '/home/divyansh/Research/Federated_Learning/train_federated/log/CASIA_1000_subjects_federated_random/20210208-090941')

	p = multiprocessing.Process(target=train_federated_one_subject_per_client, args=(config,0,class_embeddings,log_dir, None))
	p.start()
	p.join()

	for comm_round in range(config.num_comm_rounds-1):
		global_weights = np.load('./global_weights.npy', allow_pickle=True)
		p = multiprocessing.Process(target=train_federated_one_subject_per_client, args=(config, comm_round, class_embeddings,log_dir,global_weights))
		p.start()
		p.join()

	# pdb.set_trace()
	# train_federated_one_subject_per_client(config, config_file)


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("config_file", help="The path to the training configuration file",
						type=str)
	args = parser.parse_args()
	main(args)
	
