from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
# import tfwatcher

model_params = {
    '4': ([0, 0, 0, 0], [64, 128, 256, 512]),
    '10': ([0, 1, 2, 0], [64, 128, 256, 512]),
    '20': ([1, 2, 4, 1], [64, 128, 256, 512]),
    '36': ([2, 4, 8, 2], [64, 128, 256, 512]),
    '64': ([3, 8, 16, 3], [64, 128, 256, 512]),
    '92': ([4, 12, 24, 4], [64, 128, 256, 512]),
    '116': ([4, 16, 32, 4], [64, 128, 256, 512]),
}

batch_norm_params = {
    # Decay for the moving averages.
    'decay': 0.995,
    # epsilon to prevent 0s in variance.
    'epsilon': 0.001,
    # force in-place updates of mean and variance estimates
    'updates_collections': None,
    # Moving averages ends up in the trainable variables collection
    'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
}   

batch_norm_params_last = {
    'decay': 0.995,
    'epsilon': 1e-8,
    'center': False,
    'scale': True,
    'updates_collections': None,
    'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    'param_initializers': {'gamma': tf.constant_initializer(0.1)},
}

def parametric_relu(x):
    num_channels = x.shape[-1].value
    with tf.variable_scope('PRELU'):
        alpha = tf.get_variable('alpha', (1,1,1,num_channels),
                        initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
        mask = x>=0
        mask_pos = tf.cast(mask, tf.float32)
        mask_neg = tf.cast(tf.logical_not(mask), tf.float32)
        return mask_pos * x + mask_neg * alpha * x

# activation = parametric_relu
activation = lambda x: tf.keras.layers.PReLU(shared_axes=[1,2]).apply(x)
# activation = tf.nn.softplus

def se_module(input_net, ratio=16, reuse = None, scope = None):
    with tf.variable_scope(scope, 'SE', [input_net], reuse=reuse):
        h,w,c = tuple([dim.value for dim in input_net.shape[1:4]])
        assert c % ratio == 0
        hidden_units = int(c / ratio)
        squeeze = slim.avg_pool2d(input_net, [h,w], padding='VALID')
        excitation = slim.flatten(squeeze)
        excitation = slim.fully_connected(excitation, hidden_units, scope='se_fc1',
                                weights_regularizer=None,
                                # weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                weights_initializer=slim.xavier_initializer(), 
                                activation_fn=tf.nn.relu)
        excitation = slim.fully_connected(excitation, c, scope='se_fc2',
                                weights_regularizer=None,
                                # weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                weights_initializer=slim.xavier_initializer(),
                                biases_initializer=None,
                                activation_fn=tf.nn.sigmoid)        
        excitation = tf.reshape(excitation, [-1,1,1,c])
        output_net = input_net * excitation

        return output_net

def conv_module(net, num_res_layers, num_kernels, trans_kernel_size=3, trans_stride=2,
                     use_se=False, reuse=None, scope=None):
    with tf.variable_scope(scope, 'conv', [net], reuse=reuse):
        net = slim.conv2d(net, num_kernels, kernel_size=trans_kernel_size, stride=trans_stride, padding='SAME',
                weights_initializer=slim.xavier_initializer())
        shortcut = net
        for i in range(num_res_layers):
            # num_kernels_sm = int(num_kernels / 2)
            net = slim.conv2d(net, num_kernels, kernel_size=3, stride=1, padding='SAME',
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                biases_initializer=None)
            net = slim.conv2d(net, num_kernels, kernel_size=3, stride=1, padding='SAME',
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                biases_initializer=None)
            # net = slim.conv2d(net, num_kernels, kernel_size=1, stride=1, padding='SAME',
            #     weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
            #     biases_initializer=None)
            print('| ---- block_%d' % i)
            if use_se:
                net = se_module(net)
            net = net + shortcut
            shortcut = net
    return net

def inference(images, keep_probability, phase_train=True, bottleneck_layer_size=512, 
            weight_decay=0.0, reuse=tf.AUTO_REUSE, model_version=None):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        activation_fn=activation,
                        normalizer_fn=None,
                        normalizer_params=batch_norm_params):
        with tf.variable_scope('SphereNet', [images, slim.conv2d], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=phase_train):
                print('SphereNet input shape:', [dim.value for dim in images.shape])
                print('REUSE : ', reuse)
                model_version = '4' if model_version ==None else model_version
                num_layers, num_kernels = model_params[model_version]

                net = conv_module(images, num_layers[0], num_kernels[0], scope='conv1')
                print('module_1 shape:', [dim.value for dim in net.shape])

                net = conv_module(net, num_layers[1], num_kernels[1], scope='conv2')
                print('module_2 shape:', [dim.value for dim in net.shape])
                
                net = conv_module(net, num_layers[2], num_kernels[2], scope='conv3')
                print('module_3 shape:', [dim.value for dim in net.shape])

                net = conv_module(net, num_layers[3], num_kernels[3], scope='conv4')
                print('module_4 shape:', [dim.value for dim in net.shape])
                
                # conv_bottleneck_size = int(bottleneck_layer_size / 4)
                # net = conv_module(net, 0, conv_bottleneck_size, trans_stride=1, use_se=False, scope='conv5')
                # print('module_5 shape:', [dim.value for dim in net.shape])

                # net = slim.avg_pool2d(net, [net.shape[1], net.shape[2]], stride=1, padding='VALID', scope='global_pooling')
                net = slim.flatten(net)
                prelogits = slim.fully_connected(net, bottleneck_layer_size, scope='Bottleneck',
                                        # weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                        # weights_initializer=tf.constant_initializer(0.),
                                        weights_initializer=slim.xavier_initializer(), 
                                        activation_fn=None, normalizer_fn=None)
                print('prelogits shape: ', [dim.value for dim in prelogits.shape])
                # prelogits= slim.batch_norm(_prelogits, **batch_norm_params_last)
                # logits = slim.fully_connected(prelogits, num_classes, scope='Logits',
                #                         weights_initializer=slim.xavier_initializer(),
                #                         activation_fn=None, normalizer_fn=None)
                # print('logits shape: ', [dim.value for dim in logits.shape])
    return prelogits
