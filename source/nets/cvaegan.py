from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import layer_norm

batch_norm_params = {
    'decay': 0.995,
    'epsilon': 0.001,
    'updates_collections': None,
    'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
}

gf_dim = 64

def leaky_relu(x):
    return tf.maximum(0.2 * x, x)

def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2d'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
        x = tf.tile(x, [1, 1, factor, 1, factor, 1])
        x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
        return x

def padding(x, pad, pad_type='reflect'):
    if pad_type == 'zero':
        return tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
    if pad_type == 'reflect':
        return tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0,]], mode='REFLECT')
    else:
        raise ValueError('Unknown pad type: {}'.format(pad_type))

def conv(x, *args, pad=1, **kwargs):
    with slim.arg_scope([slim.conv2d], padding='VALID'):
        x = padding(x, pad)
        return slim.conv2d(x, *args, **kwargs)

def deconv(x, *args, pad=1, **kwargs):
    with slim.arg_scope([slim.conv2d_transpose], padding='VALID'):
        x = padding(x, pad)
        return slim.conv2d_transpose(x, *args, **kwargs)

def encoder(images, style_size=1, keep_prob=1.0, phase_train=True, weight_decay=0.0, reuse=None, scope='Encoder'):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        # weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.dropout, slim.batch_norm], is_training=phase_train):
                with slim.arg_scope([slim.fully_connected],
                    normalizer_fn=layer_norm, normalizer_params=None):
                    print('{} input shape:'.format(scope), [dim.value for dim in images.shape])

                    batch_size = tf.shape(images)[0]
                    k = 64


                    with tf.variable_scope('StyleEncoder'):
                        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
                            normalizer_fn=None, normalizer_params=None):
                            
                            print('-- StyleEncoder')

                            net = images

                            net = conv(net, k, 7, stride=1, pad=3, scope='conv0')
                            print('module conv0 shape:', [dim.value for dim in net.shape])

                            net = conv(net, 2*k, 4, stride=2, scope='conv1')
                            print('module conv1 shape:', [dim.value for dim in net.shape])

                            net = conv(net, 4*k, 4, stride=2, scope='conv2')
                            print('module conv2 shape:', [dim.value for dim in net.shape])
                            
                            #net = slim.avg_pool2d(net, net.shape[1:3], padding='VALID', scope='global_pool')
                            #print('module avg shape:', [dim.value for dim in net.shape])
                            net = slim.flatten(net)
                            net = slim.fully_connected(net, 128, activation_fn=None, normalizer_fn=None, scope='fc1')
                            #net = tf.reshape(net, [-1, 64, 64, style_size])
                            print('fc1 shape: ', [dim.value for dim in net.shape])
                    return net

def simple_generator(latents, keep_prob=1.0, phase_train=True, weight_decay=0.0, reuse=None, scope='Generator'):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
             activation_fn=tf.nn.relu,                                          
             normalizer_fn=slim.batch_norm,                                     
             normalizer_params=batch_norm_params,                               
             weights_initializer=tf.contrib.layers.xavier_initializer(),        
             weights_regularizer=slim.l2_regularizer(weight_decay)):            
         with tf.variable_scope(scope, [latents], reuse=reuse):                   
            with slim.arg_scope([slim.dropout, slim.batch_norm], is_training=phase_train):
                with slim.arg_scope([slim.fully_connected], normalizer_fn=layer_norm, normalizer_params=None):
                    net = latents
                    print('{} input shape: '.format(scope), [dim.value for dim in net.shape])
     
                    net = slim.fully_connected(net, 2 * 2 * 512, activation_fn=None, normalizer_fn=None, scope='fc2')
                    net = tf.reshape(net, [-1, 2, 2, 512])
                    print('module_1 shape: ', [dim.value for dim in net.shape])
                    
                    net = slim.conv2d_transpose(net, 256, kernel_size=4, stride=2, scope='deconv2')
                    print('module_2 shape: ', [dim.value for dim in net.shape])
                    net = slim.conv2d_transpose(net, 128, kernel_size=4, stride=2, scope='deconv3')
                    print('module_3 shape: ', [dim.value for dim in net.shape])

                    net = slim.conv2d_transpose(net, 64, kernel_size=4, stride=2, scope='deconv4')
                    print('module_4 shape: ', [dim.value for dim in net.shape])
                    net = slim.conv2d_transpose(net, 32, kernel_size=4, stride=2, scope='deconv5')
                    print('module_5 shape: ', [dim.value for dim in net.shape])
                    net = slim.conv2d_transpose(net, 3, kernel_size=4, stride=2, scope='deconv6', activation_fn=None, normalizer_fn=None)
                    print('module_6 shape: ', [dim.value for dim in net.shape])

                    net = tf.nn.tanh(net, name='output')
                    print('output shape: ', [dim.value for dim in net.shape])
                    return net

def generator(latents, keep_prob=1.0, phase_train=True, weight_decay=0.0, reuse=None, scope='Generator'):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
             activation_fn=tf.nn.relu,                                          
             normalizer_fn=instance_norm,                               
             weights_initializer=tf.contrib.layers.xavier_initializer(),        
             weights_regularizer=slim.l2_regularizer(weight_decay)):            
         with tf.variable_scope(scope, [latents], reuse=reuse):                   
            with slim.arg_scope([slim.dropout, slim.batch_norm], is_training=phase_train):
                with slim.arg_scope([slim.fully_connected], normalizer_fn=layer_norm, normalizer_params=None):
                    net = latents
                    print('{} input shape: '.format(scope), [dim.value for dim in net.shape])
     
                    net = slim.fully_connected(net, 2 * 2 * 512, activation_fn=None, normalizer_fn=None, scope='fc2')
                    net = tf.reshape(net, [-1, 2, 2, 512])
                    net = slim.conv2d_transpose(net, 256, kernel_size=4, stride=2, scope='deconv2')
                    print('module_2 shape: ', [dim.value for dim in net.shape])

                    print('module_1 shape: ', [dim.value for dim in net.shape])
                    net = residual_block(net, 256, name='g_r1')
                    print('module_2 shape: ', [dim.value for dim in net.shape])
                    net = residual_block(net, 256, name='g_r2')
                    print('module_3 shape: ', [dim.value for dim in net.shape])
                    net = residual_block(net, 256, name='g_r3')
                    print('module_4 shape: ', [dim.value for dim in net.shape])
                    net = residual_block(net, 256, name='g_r4')
                    print('module_5 shape: ', [dim.value for dim in net.shape])
                    net = residual_block(net, 256, name='g_r5')
                    print('module_1 shape: ', [dim.value for dim in net.shape])
                    net = slim.conv2d_transpose(net, 128, kernel_size=4, stride=2, scope='deconv3')
                    print('module_3 shape: ', [dim.value for dim in net.shape])

                    net = slim.conv2d_transpose(net, 64, kernel_size=4, stride=2, scope='deconv4')
                    print('module_4 shape: ', [dim.value for dim in net.shape])
                    net = slim.conv2d_transpose(net, 32, kernel_size=4, stride=2, scope='deconv5')
                    print('module_5 shape: ', [dim.value for dim in net.shape])
                    net = slim.conv2d_transpose(net, 3, kernel_size=4, stride=2, scope='deconv6', activation_fn=None, normalizer_fn=None)
                    print('module_6 shape: ', [dim.value for dim in net.shape])

                    net = tf.nn.tanh(net, name='output')
                    print('output_shape: ', [dim.value for dim in net.shape])
                    return net

def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
biases_initializer=None)

def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
    return scale*normalized + offset

def residual_block(x, dim, ks=3, s=1, name='res'):
    p = int((ks-1)/2)
    y = tf.pad(x, [[0,0], [p,p], [p,p], [0,0]], "REFLECT")
    y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
    y = tf.pad(tf.nn.relu(y), [[0,0], [p,p], [p,p], [0,0]], "REFLECT")
    y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
    return y+x

def normal_discriminator(images, keep_prob=1.0, phase_train=True,
            weight_decay=0.0, reuse=None, scope='Discriminator'):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        activation_fn=leaky_relu,
                        normalizer_fn=instance_norm):
        with tf.variable_scope(scope, [images], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=phase_train):

                print('{} input shape:'.format(scope), [dim.value for dim in images.shape])

                net =conv(images, 32, kernel_size=4, stride=2, scope='conv1', activation_fn=None)
                print('module_1 shape:', [dim.value for dim in net.shape])
                
                net = conv(net, 64, kernel_size=4, stride=2, scope='conv2')
                print('module_2 shape:', [dim.value for dim in net.shape])

                net = conv(net, 128, kernel_size=4, stride=2, scope='conv3')
                print('module_3 shape:', [dim.value for dim in net.shape])
 
             
                net = conv(net, 256, kernel_size=4, stride=2, scope='conv4')
                print('module_4 shape:', [dim.value for dim in net.shape])

                net = conv(net, 512, kernel_size=4, stride=2, scope='conv5')
                print('module_5 shape:', [dim.value for dim in net.shape])

                net = slim.flatten(net)
                net = slim.fully_connected(net, 1, activation_fn=None, normalizer_fn=None)
                return net
