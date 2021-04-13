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

from .. import utils as tfutils 
from .. import losses as tflosses
from .. import watcher as tfwatcher


class YichunNet:
    def __init__(self):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(gpu_options=gpu_options,
                allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(graph=self.graph, config=tf_config)
        

    def load_model(self, model_path, scope=None):
        tfutils.load_model(self.sess, model_path, scope=scope)
        self.phase_train_placeholder = self.graph.get_tensor_by_name('phase_train:0')
        self.keep_prob_placeholder = self.graph.get_tensor_by_name('keep_prob:0')
        self.inputs = self.graph.get_tensor_by_name('inputs:0')
        self.outputs = self.graph.get_tensor_by_name('prelogits:0')
        # self.outputs = self.graph.get_tensor_by_name('SphereNet/Flatten/flatten/Reshape:0')
        # self.sigma_sq = self.graph.get_tensor_by_name('sigma_sq:0')
        try:
            self.input_decoder = self.graph.get_tensor_by_name('input_decoder:0')
            self.output_decoder = self.graph.get_tensor_by_name('output_decoder:0')
        except:
            print(':: Decoder is NOT found in the graph.')
        # self.config = imp.load_source('network_config', os.path.join(model_path, 'config.py'))

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
            # result[start_idx:end_idx], sigma_sq[start_idx:end_idx] = self.sess.run([self.outputs, self.sigma_sq], feed_dict=feed_dict)
            result[start_idx:end_idx] = self.sess.run(self.outputs, feed_dict=feed_dict)
        if verbose:
            print('')
        return result

        
