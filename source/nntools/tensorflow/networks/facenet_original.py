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

import sys
import time
import imp
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from nntools import tensorflow as tftools
from .. import utils as tfutils 
from .. import losses as tflosses
from .. import watcher as tfwatcher

class FaceNetOriginal:
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
                h, w = config.image_size
                channels = config.channels
                image_batch_placeholder = tf.placeholder(tf.float32, shape=[None, h, w, channels], name='image_batch')
                label_batch_placeholder = tf.placeholder(tf.int32, shape=[None], name='label_batch')
                learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
                keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob')
                phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
                global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

                image_splits = tf.split(image_batch_placeholder, config.num_gpus)
                label_splits = tf.split(label_batch_placeholder, config.num_gpus)
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
                                # Save the first channel for testing
                                if i == 0:
                                    self.inputs = images
                                
                                # Build networks
                                if config.localization_net is not None:
                                    localization_net = imp.load_source('localization_net', config.localization_net)
                                    images_original = images
                                    images, theta = localization_net.inference(images, config.localization_output, 
                                                    phase_train_placeholder,
                                                    weight_decay = 0.0)
                                    if False:                                    
                                        images_left, images_right = tftools.tensor_ops.split_pairs(images)
                                        images_original = images_left
                                        labels_left, _ = tftools.tensor_ops.split_pairs(labels)
                                        images_interp = tftools.tensor_ops.random_interpolate(images_left, images_right)
                                        # Do not use use interploated images during testing
                                        images = tf.cond(phase_train_placeholder, lambda: images_interp, lambda: images)
                                        labels = tf.cond(phase_train_placeholder, lambda: labels_left, lambda: labels)
                                        
                                    images = tf.identity(images, name='transformed_image')
                                    if i == 0:
                                        tf.summary.image('original_image', images_original)
                                        tf.summary.image('transformed_image', images)
                                else:
                                    images = images

                                network = imp.load_source('network', config.network)
                                prelogits, _ = network.inference(images, keep_prob_placeholder, phase_train_placeholder,
                                                        bottleneck_layer_size = config.embedding_size, 
                                                        weight_decay = config.weight_decay)
                                prelogits = tf.identity(prelogits, name='prelogits')
                                # embeddings = tflosses.group_normalize(prelogits, 2, name='embeddings')
                                embeddings = tf.nn.l2_normalize(prelogits, dim=1, name='embeddings')
                                if i == 0:
                                    self.outputs = tf.identity(prelogits, name='outputs')

                                # Build all losses
                                loss_list = []

                                # Orignal Softmax
                                if 'softmax' in config.losses.keys():
                                    softmax_loss = tflosses.softmax_loss(prelogits, labels, num_classes,
                                                    weight_decay=config.weight_decay, **config.losses['softmax'])
                                    loss_list.append(softmax_loss)
                                    insert_dict('sfloss', softmax_loss)
                                # Center Loss
                                if 'center' in config.losses.keys():
                                    center_loss = tflosses.center_loss(prelogits, labels, 
                                                        num_classes, **config.losses['center'])
                                    loss_list.append(center_loss)
                                    insert_dict('ctloss', center_loss)
                                # Ring Loss
                                if 'ring' in config.losses.keys():
                                    ring_loss = tflosses.ring_loss(prelogits, **config.losses['ring'])
                                    loss_list.append(ring_loss)
                                    insert_dict('rloss', ring_loss)
                                # Decov Loss
                                if 'decov' in config.losses.keys():
                                    decov_loss = tflosses.decov_loss(prelogits, **config.losses['decov'])
                                    loss_list.append(decov_loss)
                                    insert_dict('decloss', decov_loss)
                                # Triplet Loss
                                if 'triplet' in config.losses.keys():
                                    triplet_loss = tflosses.triplet_semihard_loss(labels, embeddings, **config.losses['triplet'])
                                    loss_list.append(triplet_loss)
                                    insert_dict('loss', triplet_loss)
                                # Contrastive Loss
                                if 'contrastive' in config.losses.keys():
                                    contrastive_loss = tflosses.contrastive_loss(labels, prelogits, **config.losses['contrastive'])
                                    loss_list.append(contrastive_loss)
                                    insert_dict('loss', contrastive_loss)

                                # L2-Softmax
                                if 'cosine' in config.losses.keys():
                                    logits, cosine_loss = losses.cosine_softmax(prelogits, labels, num_classes, 
                                                            gamma=config.losses['cosine']['gamma'], 
                                                            weight_decay=config.weight_decay)
                                    loss_list.append(cosine_loss)
                                    insert_dict('closs', cosine_loss)
                                # A-Softmax
                                if 'angular' in config.losses.keys():
                                    a_cfg = config.losses['angular']
                                    angular_loss = tflosses.angular_softmax(prelogits, labels, num_classes, 
                                                            global_step, a_cfg['m'], a_cfg['lamb_min'], a_cfg['lamb_max'],
                                                            weight_decay=config.weight_decay)
                                    loss_list.append(angular_loss)
                                    insert_dict('aloss', angular_loss)
                                # AM-Softmax
                                if 'am_softmax' in config.losses.keys():
                                    amloss = tflosses.am_softmax(prelogits, labels, num_classes, 
                                                            global_step, weight_decay=config.weight_decay,
                                                            **config.losses['am_softmax'])
                                    loss_list.append(amloss)
                                    insert_dict('loss', amloss)
                                # AM-Softmax Dynamic Imprinting
                                if 'am_imprint' in config.losses.keys():
                                    ami_loss = tflosses.am_softmax_imprint(prelogits, labels, num_classes,
                                        global_step, config.weight_decay, learning_rate_placeholder, **config.losses['am_imprint'])
                                    ami_loss = tf.identity(ami_loss, name='ami_loss')
                                    loss_list.append(ami_loss)
                                    insert_dict('amiloss', ami_loss)
                                # ArcFace
                                if 'arcface' in config.losses.keys():
                                    arc_loss = tflosses.arcface_loss(prelogits, labels, num_classes, **config.losses['arcface'])
                                    arc_loss = tf.identity(arc_loss, name='arc_loss')
                                    loss_list.append(arc_loss)
                                    insert_dict('arcloss', arc_loss)

                                # Euclidean Loss
                                if 'euc' in config.losses.keys():
                                    euc_loss = tflosses.euc_loss(prelogits, labels, num_classes,
                                        global_step, config.weight_decay, **config.losses['euc'])
                                    loss_list.append(euc_loss)
                                    insert_dict('euc_loss', euc_loss)
                                # Split Loss
                                if 'split' in config.losses.keys():
                                    split_losses = tflosses.split_softmax(prelogits, labels, num_classes, 
                                                            global_step, **config.losses['split'])
                                    loss_list.append(split_losses)
                                    insert_dict('sploss', split_losses)
                                # MPS
                                if 'pair' in config.losses.keys():
                                    pair_losses = tflosses.pair_loss(prelogits, labels, num_classes, 
                                                            global_step, gamma=config.losses['pair']['gamma'],  
                                                            m=config.losses['pair']['m'],
                                                            weight_decay=config.weight_decay)
                                    loss_list.extend(pair_losses)
                                    insert_dict('loss', pair_losses[0])
                                    

                               # Collect all losses
                                reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
                                loss_list.append(reg_loss)
                                insert_dict('reg_loss', reg_loss)

                                total_loss = tf.add_n(loss_list, name='total_loss')
                                grads_split = tf.gradients(total_loss, tf.trainable_variables())
                                grads_splits.append(grads_split)



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

                train_ops = [apply_gradient_op, update_global_step_op] + update_ops
                train_op = tf.group(*train_ops)

                tf.summary.scalar('learning_rate', learning_rate_placeholder)
                summary_op = tf.summary.merge_all()

                # Initialize variables
                self.sess.run(tf.local_variables_initializer())
                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver(tf.trainable_variables())


                # Keep useful tensors
                self.image_batch_placeholder = image_batch_placeholder
                self.label_batch_placeholder = label_batch_placeholder 
                self.learning_rate_placeholder = learning_rate_placeholder 
                self.keep_prob_placeholder = keep_prob_placeholder 
                self.phase_train_placeholder = phase_train_placeholder 
                self.global_step = global_step
                self.train_op = train_op
                self.summary_op = summary_op
                


    def train(self, image_batch, label_batch, learning_rate, keep_prob):
        feed_dict = {self.image_batch_placeholder: image_batch,
                    self.label_batch_placeholder: label_batch,
                    self.learning_rate_placeholder: learning_rate,
                    self.keep_prob_placeholder: keep_prob,
                    self.phase_train_placeholder: True,}
        _, wl, sm = self.sess.run([self.train_op, tfwatcher.get_watchlist(), self.summary_op], feed_dict = feed_dict)
        step = self.sess.run(self.global_step)

        return wl, sm, step
    
    def restore_model(self, *args, **kwargs):
        trainable_variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        tfutils.restore_model(self.sess, trainable_variables, *args, **kwargs)

    def save_model(self, model_dir, global_step):
        tfutils.save_model(self.sess, self.saver, model_dir, global_step)
        

    def load_model(self, *args, **kwargs):
        tfutils.load_model(self.sess, *args, **kwargs)
        print([n.name for n in self.graph.as_graph_def().node if 'keep_prob' in n.name])
        self.phase_train_placeholder = self.graph.get_tensor_by_name('InceptionResnetV1/phase_train:0')
        self.keep_prob_placeholder = self.graph.get_tensor_by_name('InceptionResnetV1/InceptionResnetV1/Logits/Dropout/cond/dropout/keep_prob:0')
        # self.keep_prob_placeholder = self.graph.get_tensor_by_name('InceptionResnetV1/keep_prob:0')
        self.inputs = self.graph.get_tensor_by_name('InceptionResnetV1/input:0')
        self.outputs = self.graph.get_tensor_by_name('InceptionResnetV1/InceptionResnetV1/Bottleneck/MatMul:0')
        #self.intermediate = self.graph.get_tensor_by_name('InceptionResnetV1/Conv2d_4b_3x3/Relu:0')
    
    def extract_intermediate_feature(self, images, batch_size, proc_func=None, verbose=False):
        num_images = images.shape[0] if type(images)==np.ndarray else len(images)
        h,w,c = tuple(self.intermediate.shape[1:])
        result = np.ndarray((num_images, h,w,c), dtype=np.float32)
        start_time = time.time()
        for start_idx in range(0, num_images, batch_size):
            if verbose:
                elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))
                sys.stdout.write('# of images: %d Current image: %d Elapsed time: %s \t\r' 
                    % (num_images, start_idx, elapsed_time))
            end_idx = min(num_images, start_idx + batch_size)
            inputs = images[start_idx:end_idx]
            inputs = proc_func(inputs) if proc_func else inputs
            feed_dict = {self.inputs: inputs,
                        self.phase_train_placeholder: False,
                        self.keep_prob_placeholder: 1.0}
            result[start_idx:end_idx] = self.sess.run(self.intermediate, feed_dict=feed_dict)
        return result

    def extract_feature(self, images, batch_size, proc_func=None, verbose=False):
        num_images = images.shape[0] if type(images)==np.ndarray else len(images)
        num_features = self.outputs.shape[1]
        result = np.ndarray((num_images, num_features), dtype=np.float32)
        start_time = time.time()
        for start_idx in range(0, num_images, batch_size):
            if verbose:
                elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))
                sys.stdout.write('# of images: %d Current image: %d Elapsed time: %s \t\r' 
                    % (num_images, start_idx, elapsed_time))
            end_idx = min(num_images, start_idx + batch_size)
            inputs = images[start_idx:end_idx]
            inputs = proc_func(inputs) if proc_func else inputs
            feed_dict = {self.inputs: inputs,
                        self.phase_train_placeholder: False,
                        self.keep_prob_placeholder: 1.0}
            result[start_idx:end_idx] = self.sess.run(self.outputs, feed_dict=feed_dict)
        return result

        
