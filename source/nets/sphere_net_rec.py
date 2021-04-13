from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nntools.tensorflow import watcher as tfwatcher

model_params = {
    '4': ([0, 0, 0, 0], [64, 128, 256, 512]),
    '10': ([0, 1, 2, 0], [64, 128, 256, 512]),
    '20sm': ([1, 2, 4, 1], [32, 64, 128, 256]),
    '20': ([1, 2, 4, 1], [64, 128, 256, 512]),
    '36': ([2, 4, 8, 2], [64, 128, 256, 512]),
    '64': ([3, 8, 16, 3], [64, 128, 256, 512]),
    '92': ([4, 12, 24, 4], [64, 128, 256, 512]),
    '116': ([4, 16, 32, 4], [64, 128, 256, 512]),
}

batch_norm_params = {
    'decay': 0.995,
    'epsilon': 0.001,
    'center': False,
    'scale': True,
    'is_training': False, # Frozen during training
    'updates_collections': None,
    'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
}   

batch_norm_params_last = {
    'decay': 0.995,
    'epsilon': 0.001,
    'center': True,
    'scale': False,
    'is_training': False, # Frozen during training
    'updates_collections': None,
    'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    'param_initializers': {'gamma': tf.constant_initializer(0.1)},
}

batch_norm_params_module = {
    'decay': 0.995,
    'epsilon': 0.001,
    'center': True,
    'scale': True,
    'is_training': False, # Frozen during training
    'updates_collections': None,
    'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
}

batch_norm_params_sigma = {
    'decay': 0.995,
    'epsilon': 0.001,
    'center': False,
    'scale': False,
    'is_training': False, # Frozen during training
    'updates_collections': None,
    'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    'param_initializers': {'gamma': tf.constant_initializer(1e-4), 'beta': tf.constant_initializer(-7.)},
    # 'param_regularizers': {'gamma': slim.l2_regularizer(1.0)}
}

def parametric_relu(x):
    num_channels = x.shape[-1].value
    with tf.variable_scope('p_re_lu'):
        alpha = tf.get_variable('alpha', (1,1,num_channels),
                        initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
        return tf.nn.relu(x) + alpha * tf.minimum(0.0, x)

def scale_and_shift(x, gamma_init=1.0, beta_init=0.0):
    num_channels = x.shape[-1].value
    with tf.variable_scope('scale_and_shift'):
        gamma = tf.get_variable('alpha', (),
                        initializer=tf.constant_initializer(gamma_init),
                        regularizer=slim.l2_regularizer(0.0),
                        dtype=tf.float32)
        beta = tf.get_variable('gamma', (),
                        initializer=tf.constant_initializer(beta_init),
                        dtype=tf.float32)
        x = gamma * x +  beta

        return x   

def ratio_dropout(x, min_keep_prob, max_keep_prob, is_training, scope='ratio_dropout'):
    with tf.name_scope('ratio_dropout'):
        batch_size = tf.shape(x)[0]
        ndims = x.shape.ndims
        
        keep_ratios = tf.random_uniform((batch_size,), 
            min_keep_prob, max_keep_prob, dtype=tf.float32)
        keep_ratios = tf.reshape(keep_ratios, [-1] + [1]*(ndims-1))

        mask = tf.random_uniform(tf.shape(x), -1., 0., dtype=tf.float32)
        mask = tf.cast(tf.greater(mask + keep_ratios, 0.), dtype=tf.float32)
        return tf.cond(is_training, lambda: x * mask, lambda: x)
    
    
    

activation = parametric_relu
# activation = lambda x: tf.keras.layers.PReLU(shared_axes=[1,2]).apply(x)
# activation = tf.nn.relu
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
                                activation_fn=tf.nn.sigmoid)        
        excitation = tf.reshape(excitation, [-1,1,1,c])
        output_net = input_net * excitation

        return output_net

def conv_module(net, num_res_layers, num_kernels, trans_kernel_size=3, trans_stride=2,
                     use_se=False, reuse=None, scope=None):
    with tf.variable_scope(scope, 'conv', [net], reuse=reuse):
        net = slim.conv2d(net, num_kernels, kernel_size=trans_kernel_size, stride=trans_stride, padding='SAME',
                weights_initializer=slim.xavier_initializer()) #, normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params)
        shortcut = net
        for i in range(num_res_layers):
            net = slim.conv2d(net, num_kernels, kernel_size=3, stride=1, padding='SAME',
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                biases_initializer=None)
            net = slim.conv2d(net, num_kernels, kernel_size=3, stride=1, padding='SAME',
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                biases_initializer=None)
            print('| ---- block_%d' % i)
            if use_se:
                net = se_module(net)
            net = net + shortcut
            shortcut = net
    return net

def inference(images, keep_probability, phase_train=True, bottleneck_layer_size=512, 
            weight_decay=0.0, reuse=None, model_version='64'):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=None, #slim.batch_norm,
                        normalizer_params=None, #batch_norm_params,
                        activation_fn=activation):
        with tf.variable_scope('SphereNet', [images], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=phase_train):
                print('SphereNet input shape:', [dim.value for dim in images.shape])
                
                model_version = '64' if model_version ==None else model_version
                num_layers, num_kernels = model_params[model_version]

                convs = []

                net = conv_module(images, num_layers[0], num_kernels[0], scope='conv1')
                print('module_1 shape:', [dim.value for dim in net.shape])
                #net = ratio_dropout(net, 0.7, 1.0, phase_train, 'dropout1')


                net = conv_module(net, num_layers[1], num_kernels[1], scope='conv2')
                print('module_2 shape:', [dim.value for dim in net.shape])
                #net = ratio_dropout(net, 0.7, 1.0, phase_train, 'dropout2')
                convs.append(net)
                
                net = conv_module(net, num_layers[2], num_kernels[2], scope='conv3')
                print('module_3 shape:', [dim.value for dim in net.shape])
                #net = ratio_dropout(net, 0.7, 1.0, phase_train, 'dropout3')
                convs.append(net)

                net = conv_module(net, num_layers[3], num_kernels[3], use_se=False, scope='conv4')
                print('module_4 shape:', [dim.value for dim in net.shape])
                #net = ratio_dropout(net, 0.7, 1.0, phase_train, 'dropout4')
                convs.append(net)

                # net = ratio_dropout(net, 0.3, 1.0, phase_train, 'dropout1')
                net_ = net
                net = slim.flatten(net)

                # net = slim.fully_connected(net_, 1024, scope='fc1')[0]

                prelogits = slim.fully_connected(net, bottleneck_layer_size, scope='Bottleneck',
                                        # weights_initializer=tf.truncated_normal_initializer(stddev=1e-1),
                                        # weights_initializer=tf.constant_initializer(0.),
                                        weights_initializer=slim.xavier_initializer(),
                                        biases_initializer=None,
                                        # normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params_last,
                                        activation_fn=None)
                
                # scale = tf.get_variable(name='scale', shape=(), trainable=True, 
                #                         initializer=tf.constant_initializer(1.0))
                # prelogits = scale * prelogits
                prelogits = tf.nn.l2_normalize(prelogits, axis=1)
                # prelogits = tf.nn.tanh(prelogits)
                

                # for i,conv in enumerate(convs):
                #     net = slim.flatten(conv)
                #      net = slim.fully_connected(net, 128, scope='fc_conv{}'.format(i),
                #         normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params, activation_fn=tf.nn.relu)
                #     convs[i] = slim.flatten(slim.avg_pool2d(convs[i], convs[i].shape[1:3]))
                # net = tf.concat(convs, axis=1)

                # with tf.variable_scope('UncertaintyModule'):

                #     dec = 5e-4

                #     if False:
                #         net = slim.conv2d(net_, 512, kernel_size=1, stride=1, padding='SAME',
                #             normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params_module,
                #             weights_regularizer=slim.l2_regularizer(dec),
                #             activation_fn=tf.nn.relu)
                #     else:
                #         net = net_
                #         net_ = slim.flatten(net_)

                #     # net = slim.avg_pool2d(net, net.shape[1:3])

                #     net = slim.flatten(net)

                #     if True:
                #         net = slim.fully_connected(net, 512, scope='fc1',
                #             normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params_module, 
                #             weights_regularizer=slim.l2_regularizer(dec),
                #             activation_fn=tf.nn.relu)
                #     # net = slim.dropout(net, keep_prob=0.5)

                #     log_sigma_sq = slim.fully_connected(net, bottleneck_layer_size, scope='fc_log_sigma_sq',
                #                             # weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                #                             weights_initializer=slim.xavier_initializer(),
                #                             weights_regularizer=slim.l2_regularizer(dec),
                #                             # weights_initializer=tf.constant_initializer(0.),
                #                             # biases_initializer=tf.constant_initializer(-4.),
                #                             normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params_sigma,
                #                             activation_fn=None)
                #     if True:
                #         log_sigma_sq = scale_and_shift(log_sigma_sq, 1e-4, -7.0)
                #     elif False:
                #         bias = tf.get_variable(name='bias', shape=(), trainable=True, 
                #                                initializer=tf.constant_initializer(-7.))
                #         log_sigma_sq = log_sigma_sq + bias

                #     if False:
                #         closs = 1e4 * tf.reduce_mean(tf.exp(log_sigma_sq), name='loss_c')
                #         tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, closs)
                #         tfwatcher.insert('closs', closs)
                #         tf.summary.scalar('closs', closs)

                #     l_avg = tf.reduce_mean(log_sigma_sq)
                #     l_std = tf.reduce_mean(tf.sqrt(tf.nn.moments(log_sigma_sq, axes=[0])[1]))
                #     tfwatcher.insert('avg', l_avg)
                #     tfwatcher.insert('std', l_std)
                #     tf.summary.scalar('avg', l_avg)
                #     tf.summary.scalar('std', l_std)
                #     log_sigma_sq = tf.log(1e-6 + tf.exp(log_sigma_sq))
                    # log_sigma_sq = tf.tile(log_sigma_sq, [1, bottleneck_layer_size])

    return prelogits




def leaky_relu(x):
    return tf.maximum(0.2*x, x)


def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
        x = tf.tile(x, [1, 1, factor, 1, factor, 1])
        x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
        return x

def padding(x, pad, pad_type='reflect'):
    if pad_type == 'zero' :
        return tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
    if pad_type == 'reflect' :
        return tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')
    else:
        raise ValueError('Unknown pad type: {}'.format(pad_type))

def conv(x, *args, pad=1, **kwargs):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], padding='VALID'):
        x = padding(x, pad)
        return slim.conv2d(x, *args, **kwargs)

def deconv(x, *args, pad=1, **kwargs):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], padding='VALID'):
        x = padding(x, pad)
        return slim.conv2d_transpose(x, *args, **kwargs)

def decoder(latent, keep_probability, phase_train=True, weight_decay=0.0, 
                            scope='Decoder', reuse=None, model_version=None):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=None, #slim.batch_norm,
                        normalizer_params=None, #batch_norm_params,
                        activation_fn=activation):
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=phase_train):
                print('Decoder input shape:', [dim.value for dim in latent.shape])
                

                net = slim.fully_connected(latent, 7*6*512, scope='fc1')
                net = tf.reshape(net, [-1, 7, 6, 512])

                net = upscale2d(net, 2)
                net = conv(net, 256, 5, stride=1, pad=2, scope='conv1')
                print('module_1 shape:', [dim.value for dim in net.shape])

                net = upscale2d(net, 2)
                net = conv(net, 128, 5, stride=1, pad=2, scope='conv2')
                print('module_2 shape:', [dim.value for dim in net.shape])

                net = upscale2d(net, 2)
                net = conv(net, 64, 5, stride=1, pad=2, scope='conv3')
                print('module_3 shape:', [dim.value for dim in net.shape])

                net = upscale2d(net, 2)
                net = conv(net, 3, 5, stride=1, pad=2, activation_fn=tf.nn.tanh, scope='conv4')
                print('module_4 shape:', [dim.value for dim in net.shape])


                return net



