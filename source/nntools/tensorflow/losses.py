import os
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from nntools.tensorflow import watcher as tfwatcher
from nntools.tensorflow.metric_loss_ops import masked_minimum, masked_maximum, triplet_semihard_loss, npairs_loss, lifted_struct_loss



# @tf.custom_gradient
# def pseudo_loss(x, grad):
#     def pseudo_grad(dy):
#         return grad, None
#     return tf.constant(0.), pseudo_grad

def pair_shuffle(x):
    num_features = x.shape[1].value
    x = tf.reshape(x, [-1, 2, num_features])
    x1, x2 = x[:,0], x[:,1]
    select = tf.random_uniform(tf.shape(x1)) >= 0.5
    x1_new = tf.where(select, x1, x2)
    x2_new = tf.where(select, x2, x1)
    x_new = tf.reshape(tf.stack([x1_new, x2_new], axis=1), [-1, num_features])
    return x_new
    
def group_shuffle(x, groups=4):
    num_features = x.shape[1].value
    input_splits = tf.split(x, groups, axis=0)
    x = tf.reshape(x, [-1, 2, num_features])
    x1, x2 = x[:,0], x[:,1]
    select = tf.random_uniform(tf.shape(x1)) >= 0.5
    x1_new = tf.where(select, x1, x2)
    x2_new = tf.where(select, x2, x1)
    x_new = tf.reshape(tf.stack([x1_new, x2_new], axis=1), [-1, num_features])
    return x_new
    

def group_normalize(x, groups, name=None):
    import math
    num_features = x.shape[1].value
    assert num_features % groups == 0
    gdim = int(num_features / groups)
    x_normed = tf.nn.l2_normalize(tf.reshape(x, [-1, groups, gdim]), dim=2) 
    x_normed = tf.reshape(x_normed, [-1, num_features], name=name)
    return x_normed

def batch_norm(x, center=True, scale=True, name=None):
    batch_norm_params = {
        'decay': 0.995,
        'epsilon': 1e-8,
        'center': center,
        'scale': scale,
        'updates_collections': None,
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
        'param_initializers': {'gamma': tf.constant_initializer(0.1)},
    }
    x_normed = slim.batch_norm(x, **batch_norm_params)    
    return tf.identity(x_normed, name=name)

def normalize_embeddings(features, normalization, name=None):
    if normalization == 'l2':
       return tf.nn.l2_normalize(features, dim=1, name=name)
    if normalization == 'batch':
        return batch_norm(features, name=name)
    elif normalization == 'std_batch':
        return batch_norm(features, False, False, name)
    elif normalization == 'scale_batch':
        return batch_norm(features, False, True, name)
    elif normalization.startswith('grpl2'):
        gdim = normalization.split(':')[1]
        gdim = int(grp_size)
        return group_normalize(x, gdim, name)      
    else:
        raise ValueError('Unkown normalization for embeddings: {}'.format(normalization))

def euclidean_distance(X, Y, sqrt=False):
    '''Compute the distance between each X and Y.

    Args: 
        X: a (m x d) tensor
        Y: a (d x n) tensor
    
    Returns: 
        diffs: an m x n distance matrix.
    '''
    with tf.name_scope('EuclideanDistance'):
        XX = tf.reduce_sum(tf.square(X), 1, keep_dims=True)
        YY = tf.reduce_sum(tf.square(Y), 0, keep_dims=True)
        XY = tf.matmul(X, Y)
        diffs = XX + YY - 2*XY
        diffs = tf.maximum(0.0, diffs)
        if sqrt == True:
            diffs = tf.sqrt(diffs)
    return diffs


def mahalanobis_distance(X, Y, sigma_sq_inv, sqrt=False):
    '''Compute the distance between each X and Y.

    Args: 
        X: a (m x d) tensor
        Y: a (d x n) tensor
        sigma_sq: a (m, d) tensor
    
    Returns: 
        diffs: an m x n distance matrix.
    '''
    with tf.name_scope('MahalanobisDistance'):
        XX = tf.reduce_sum(tf.square(X) * sigma_sq_inv, 1, keep_dims=True)
        YY = tf.matmul(sigma_sq_inv, tf.square(Y))
        XY = tf.matmul(X * sigma_sq_inv, Y)

        diffs = XX + YY - 2*XY
        if sqrt == True:
            diffs = tf.sqrt(tf.maximum(0.0, diffs))
    return diffs


def uncertain_distance(X, Y, sigma_sq_X, sigma_sq_Y, mean=False):
    with tf.name_scope('UncertainDistance'):
        if mean:
            D = X.shape[1].value

            Y = tf.transpose(Y)
            XX = tf.reduce_sum(tf.square(X), 1, keep_dims=True)
            YY = tf.reduce_sum(tf.square(Y), 0, keep_dims=True)
            XY = tf.matmul(X, Y)
            diffs = XX + YY - 2*XY

            sigma_sq_Y = tf.transpose(sigma_sq_Y)
            sigma_sq_X = tf.reduce_mean(sigma_sq_X, axis=1, keep_dims=True)
            sigma_sq_Y = tf.reduce_mean(sigma_sq_Y, axis=0, keep_dims=True)
            sigma_sq_fuse = sigma_sq_X + sigma_sq_Y

            diffs = diffs / (1e-8 + sigma_sq_fuse) + D * tf.log(sigma_sq_fuse)

            return diffs
        else:
            D = X.shape[1].value
            X = tf.reshape(X, [-1, 1, D])
            Y = tf.reshape(Y, [1, -1, D])
            sigma_sq_X = tf.reshape(sigma_sq_X, [-1, 1, D])
            sigma_sq_Y = tf.reshape(sigma_sq_Y, [1, -1, D])
            sigma_sq_fuse = sigma_sq_X + sigma_sq_Y
            diffs = tf.square(X-Y) / (1e-10 + sigma_sq_fuse) + tf.log(sigma_sq_fuse)
            return tf.reduce_sum(diffs, axis=2)


def softmax_loss(prelogits, label, num_classes, weight_decay, scope='SoftmaxLoss', reuse=None):
    num_features = prelogits.shape[1].value
    batch_size = tf.shape(prelogits)[0]
    with tf.variable_scope(scope, reuse=reuse):
        weights = tf.get_variable('weights', shape=(num_classes, num_features),
                regularizer=slim.l2_regularizer(weight_decay),
                initializer=slim.xavier_initializer(uniform=False),
                trainable = True,
                dtype=tf.float32)
        biases = tf.get_variable('biases', shape=(1, num_classes),
                initializer=tf.constant_initializer(0.),
                trainable = False,
                dtype=tf.float32)

        if False:
            bn_params = {
                'decay': 0.995,
                'epsilon': 1e-8,
                'center': True,
                'scale': True,
                'trainable': True,
                'is_training': True,
                'param_initializers': {'gamma': tf.constant_initializer(0.01)},
            }
            with slim.arg_scope([slim.fully_connected], normalizer_fn=None):#slim.batch_norm):
                weights = slim.fully_connected(weights, num_features+1, 
                    activation_fn=None, scope='fc1')
                # weights = slim.fully_connected(weights, num_features, activation_fn=None, 
                #     normalizer_fn=None, scope='fc2')
                weights, biases = weights[:,:num_features], tf.transpose(weights[:,num_features:])

            # Shuffle!!!
            prelogits = pair_shuffle(prelogits)

        logits = tf.matmul(prelogits, tf.transpose(weights)) + biases
        # logits = - euclidean_distance(prelogits, tf.transpose(weights))

        loss =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
            labels=label, logits=logits), name='loss')

        if False and weights not in tf.trainable_variables():
            alpha = 0.01
            # Update centers
            unique_label, unique_idx, unique_count = tf.unique_with_counts(label)
            appear_times = tf.gather(unique_count, unique_idx)
            appear_times = tf.reshape(appear_times, [-1, 1])

            diff_centers = tf.gather(weights, label)
            diff_centers = diff_centers / tf.cast(appear_times, tf.float32)
            diff_centers = alpha * diff_centers
            centers_update_op = tf.scatter_sub(weights, label, diff_centers)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, centers_update_op)

    return loss


def center_loss(features, labels, num_classes, alpha=0.5, coef=0.05, scope='CenterLoss', reuse=None):
    num_features = features.shape[1].value
    batch_size = tf.shape(features)[0]
    with tf.variable_scope(scope, reuse=reuse):
        centers = tf.get_variable('centers', shape=(num_classes, num_features),
                # initializer=slim.xavier_initializer(),
                initializer=tf.truncated_normal_initializer(stddev=0.1),
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.TRAINABLE_VARIABLES],
                dtype=tf.float32)

        centers_batch = tf.gather(centers, labels)
        diff_centers = centers_batch - features

        loss = coef * 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(diff_centers), axis=1), name='center_loss')

        # Update centers
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])
        diff_centers = diff_centers / tf.cast(appear_times, tf.float32)
        diff_centers = alpha * diff_centers
        centers_update_op = tf.scatter_sub(centers, labels, diff_centers)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, centers_update_op)

        return loss


def ring_loss(features, coef=0.01, scope='RingLoss', reuse=None):
    num_features = features.shape[1].value
    batch_size = tf.shape(features)[0]
    with tf.variable_scope(scope, reuse=reuse):
        R = tf.get_variable('R', shape=(),
                initializer=tf.constant_initializer(1.0),
                trainable=True, dtype=tf.float32)

        # use averaging norm instead
        R = tf.reduce_mean(tf.norm(features, axis=1))
        loss = coef * 0.5 * tf.reduce_mean(tf.square(tf.norm(features, axis=1) - R), name='ring_loss')

        return loss

def decov_loss(features, coef=0.01, scope='DecovLoss'):
    num_features = features.shape[1].value
    batch_size = tf.shape(features)[0]
    with tf.variable_scope(scope):
        cov = tf.square(tf.matmul(tf.transpose(features), features))
        nondiag = tf.logical_not(tf.eye(batch_size, dtype=tf.bool))
        loss = coef * tf.reduce_mean(tf.boolean_mask(cov,nondiag), name='decov_loss')

        return loss

def cosine_softmax(prelogits, label, num_classes, weight_decay, gamma=16.0, reuse=None):
    
    nrof_features = prelogits.shape[1].value
    
    with tf.variable_scope('Logits', reuse=reuse):
        weights = tf.get_variable('weights', shape=(nrof_features, num_classes),
                regularizer=slim.l2_regularizer(weight_decay),
                initializer=slim.xavier_initializer(),
                # initializer=tf.truncated_normal_initializer(stddev=0.1),
                dtype=tf.float32)
        alpha = tf.get_variable('alpha', shape=(),
                regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(1.00),
                trainable=True,
                dtype=tf.float32)

        weights_normed = tf.nn.l2_normalize(weights, dim=0)
        prelogits_normed = tf.nn.l2_normalize(prelogits, dim=1)

        if gamma == 'auto':
            gamma = tf.nn.softplus(alpha)
        else:
            assert type(gamma) == float
            gamma = tf.constant(gamma)

        logits = gamma * tf.matmul(prelogits_normed, weights_normed)


    cross_entropy =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
            labels=label, logits=logits), name='cross_entropy')

    tf.summary.scalar('gamma', gamma)
    tf.add_to_collection('watch_list', ('gamma', gamma))

    return logits, cross_entropy

def norm_loss(prelogits, alpha, reuse=None):
    with tf.variable_scope('NormLoss', reuse=reuse):
        sigma = tf.get_variable('sigma', shape=(),
            # regularizer=slim.l2_regularizer(weight_decay),
            initializer=tf.constant_initializer(0.1),
            trainable=True,
            dtype=tf.float32)
    prelogits_norm = tf.reduce_sum(tf.square(prelogits), axis=1)
    # norm_loss = alpha * tf.square(tf.sqrt(prelogits_norm) - sigma)
    norm_loss = alpha * prelogits_norm
    norm_loss = tf.reduce_mean(norm_loss, axis=0, name='norm_loss')

    # tf.summary.scalar('sigma', sigma)
    # tf.add_to_collection('watch_list', ('sigma', sigma))
    return norm_loss

def angular_softmax(prelogits, label, num_classes, global_step, 
            m, lamb_min, lamb_max, weight_decay, reuse=None):
    num_features = prelogits.shape[1].value
    batch_size = tf.shape(prelogits)[0]
    lamb_min = lamb_min
    lamb_max = lamb_max
    lambda_m_theta = [
        lambda x: x**0,
        lambda x: x**1,
        lambda x: 2.0*(x**2) - 1.0,
        lambda x: 4.0*(x**3) - 3.0*x,
        lambda x: 8.0*(x**4) - 8.0*(x**2) + 1.0,
        lambda x: 16.0*(x**5) - 20.0*(x**3) + 5.0*x
    ]

    with tf.variable_scope('AngularSoftmax', reuse=reuse):
        weights = tf.get_variable('weights', shape=(num_features, num_classes),
                regularizer=slim.l2_regularizer(1e-4),
                initializer=slim.xavier_initializer(uniform=False),
                # initializer=tf.random_normal_initializer(stddev=0.01),
                trainable=True,
                dtype=tf.float32)
        lamb = tf.get_variable('lambda', shape=(),
                initializer=tf.constant_initializer(lamb_max),
                trainable=False,
                dtype=tf.float32)
        prelogits_norm  = tf.sqrt(tf.reduce_sum(tf.square(prelogits), axis=1, keep_dims=True))
        weights_normed = tf.nn.l2_normalize(weights, dim=0)
        prelogits_normed = tf.nn.l2_normalize(prelogits, dim=1)

        # Compute cosine and phi
        cos_theta = tf.matmul(prelogits_normed, weights_normed)
        cos_theta = tf.minimum(1.0, tf.maximum(-1.0, cos_theta))
        theta = tf.acos(cos_theta)
        cos_m_theta = lambda_m_theta[m](cos_theta)
        k = tf.floor(m*theta / 3.14159265)
        phi_theta = tf.pow(-1.0, k) * cos_m_theta - 2.0 * k

        cos_theta = cos_theta * prelogits_norm
        phi_theta = phi_theta * prelogits_norm

        lamb_new = tf.maximum(lamb_min, lamb_max/(1.0+0.1*tf.cast(global_step, tf.float32)))
        update_lamb = tf.assign(lamb, lamb_new)
        
        # Compute loss
        with tf.control_dependencies([update_lamb]):
            label_dense = tf.one_hot(label, num_classes, dtype=tf.float32)

            logits = cos_theta
            logits -= label_dense * cos_theta * 1.0 / (1.0+lamb)
            logits += label_dense * phi_theta * 1.0 / (1.0+lamb)
            
            cross_entropy =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
                labels=label, logits=logits), name='cross_entropy')

        tf.add_to_collection('watch_list', ('lamb', lamb))

    return cross_entropy



def am_softmax(prelogits, label, num_classes, global_step, weight_decay, 
                scale=16.0, m=10.0, alpha=None, reduce_mean=True, scope='AM-Softmax', reuse=None):
    ''' Tensorflow implementation of AM-Sofmax, proposed in:
        F. Wang, W. Liu, H. Liu, and J. Cheng. Additive margin softmax for face veriﬁcation. arXiv:1801.05599, 2018.
    '''
    num_features = prelogits.shape[1].value
    batch_size = tf.shape(prelogits)[0]
    with tf.variable_scope(scope, reuse=reuse):
        weights = tf.get_variable('weights', shape=(num_features, num_classes),
                regularizer=slim.l2_regularizer(weight_decay),
                # initializer=slim.xavier_initializer(),
                initializer=tf.random_normal_initializer(stddev=0.01),
                trainable=True,
                dtype=tf.float32)
        _scale = tf.get_variable('scale', shape=(),
                regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(1.00),
                trainable=True,
                dtype=tf.float32)

        tf.add_to_collection('classifier_weights', weights)

        # Normalizing the vecotors
        weights_normed = tf.nn.l2_normalize(weights, dim=0)
        prelogits_normed = tf.nn.l2_normalize(prelogits, dim=1)
        # prelogits_normed = prelogits

        # Label and logits between batch and examplars

        label_mat = tf.one_hot(label, num_classes, dtype=tf.float32)
        label_mask_pos = tf.cast(label_mat, tf.bool)
        label_mask_neg = tf.logical_not(label_mask_pos)

        dist_mat = tf.matmul(prelogits_normed, weights_normed)
        logits_pos = tf.boolean_mask(dist_mat, label_mask_pos)
        logits_neg = tf.boolean_mask(dist_mat, label_mask_neg)


        if scale == 'auto':
            # Automatic learned scale
            scale = tf.log(tf.exp(1.0) + tf.exp(_scale))
        else:
            # Assigned scale value
            assert type(scale) == float
            scale = tf.constant(scale)

        # Losses
        _logits_pos = tf.reshape(logits_pos, [batch_size, -1])
        _logits_neg = tf.reshape(logits_neg, [batch_size, -1])

        _logits_pos = _logits_pos * scale
        _logits_neg = _logits_neg * scale
        _logits_neg = tf.reduce_logsumexp(_logits_neg, axis=1)[:,None]

        loss = tf.nn.relu(m + _logits_neg - _logits_pos)
        if reduce_mean:
            loss = tf.reduce_mean(loss, name='am_softmax')

        # Analysis
        tfwatcher.insert('scale', scale)

    return loss

def squared_hinge_loss_with_cosine_distance(prelogits, class_embedding, global_step,
                scale=16.0, m=0.9, alpha=None, reduce_mean=True, scope='Squared_Hinge_Loss', reuse=None):

    num_features = prelogits.shape[1].value
    batch_size = tf.shape(prelogits)[0]
    with tf.variable_scope(scope, reuse=reuse):

        # Normalizing the vecotors
        weights_normed = tf.nn.l2_normalize(class_embedding , dim=0)
        prelogits_normed = tf.nn.l2_normalize(prelogits, dim=1)
        # prelogits_normed = prelogits

        # Label and logits between batch and examplars

        dist_vec = tf.matmul(prelogits_normed, weights_normed)

        # assert type(scale) == float
        # scale = tf.constant(scale)

        # Losses
        logits_pos = tf.reshape(dist_vec, [batch_size, -1])

        # _logits_pos = _logits_pos * scale

        loss = tf.square(tf.nn.relu(m - logits_pos))
        if reduce_mean:
            loss = tf.reduce_mean(loss, name='Squared_Hinge_Loss')

        # Analysis
        # tfwatcher.insert('scale', scale)

    return loss

def spreadout_regularizer(weights, global_step, margin=-0.3, reduce_mean=True, scope='spreadout_regularizer', reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        num_classes = weights.shape[0].value
        similarity_matrix = tf.matmul(weights, weights, transpose_b=True)

        mask_non_diag = tf.logical_not(tf.cast(tf.eye(num_classes), tf.bool))
        dist_neg = tf.boolean_mask(similarity_matrix, mask_non_diag)

        loss = tf.square(tf.nn.relu(margin + dist_neg))

        if(reduce_mean):
            loss = tf.reduce_mean(loss, name='spreadout_regularizer')

    return loss

def arcface_loss(embedding, labels, num_classes, s=64., m=0.5, mm=0.35):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param num_classes: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    num_features = embedding.shape[1].value
    batch_size = tf.shape(embedding)[0]
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    # mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    w_init = None
    with tf.variable_scope('arcface_loss'):
        weights = tf.get_variable(name='weights', shape=(num_features, num_classes),                  
                initializer=tf.random_normal_initializer(0.01), 
                trainable=True,
                dtype=tf.float32)

        # Normalize scale to be positive if automatic
        if s == 'auto':
            s_ = tf.get_variable('s_', shape=(),
                regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(1.0),
                trainable=True,
                dtype=tf.float32)
            s = tf.log(tf.exp(0.0) + tf.exp(s_))
            tfwatcher.insert('s', s)
        else:
            assert type(s) in [float, int]
            s = tf.constant(s)

        # inputs and weights norm
        embedding = tf.nn.l2_normalize(embedding, axis=1)
        weights = tf.nn.l2_normalize(weights, axis=0)
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2 + 1e-6, name='sin_t')
        cos_mt = s * (cos_t * cos_m - sin_t * sin_m)

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        # cond_v = cos_t - threshold
        # cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)
        cond = tf.greater(cos_t, threshold)

        keep_vals = s * (cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_vals)

        mask = tf.one_hot(labels, depth=num_classes, name='one_hot_mask')
        mask = tf.cast(mask, tf.bool)
        # inv_mask = tf.subtract(1., mask, name='inverse_mask')

        s_cos_t = s * cos_t

        logits = tf.where(mask, cos_mt_temp, s_cos_t) # inv_mask * s_cos_t + mask * cos_mt_temp

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return loss

def softmax_cross_entropy_with_logits(logits, y):
    try:
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = y))
    except:
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, targets = y))


def am_softmax_imprint(prelogits, label, num_classes, global_step, weight_decay, learning_rate,
                scale=16.0, m=1.0, alpha='auto', multi_lr=False, weights_target=None, label_target=None, reuse=None):
    ''' Variant of AM-Softmax where weights are dynamically imprinted. '''
    num_features = prelogits.shape[1].value
    batch_size = tf.shape(prelogits)[0]
    with tf.variable_scope('AM-Softmax', reuse=reuse):
        weights = tf.get_variable('weights', shape=(num_classes, num_features),
                initializer=slim.xavier_initializer(),
                # initializer=tf.truncated_normal_initializer(stddev=0.0),
                # initializer=tf.constant_initializer(0),
                trainable=False,
                dtype=tf.float32)
        _scale = tf.get_variable('_scale', shape=(),
                regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(0.0),
                trainable=True,
                dtype=tf.float32)

        tf.add_to_collection('classifier_weights', weights)


        # Normalizing the vecotors
        prelogits_normed = tf.nn.l2_normalize(prelogits, dim=1)
        weights_normed = tf.nn.l2_normalize(weights, dim=1)

        # Label and logits between batch and examplars
        label_mat_glob = tf.one_hot(label, num_classes, dtype=tf.float32)
        label_mask_pos_glob = tf.cast(label_mat_glob, tf.bool)
        label_mask_neg_glob = tf.logical_not(label_mask_pos_glob)

        logits_glob = tf.matmul(prelogits_normed, tf.transpose(weights_normed))
        # logits_glob = -0.5 * euclidean_distance(prelogits_normed, tf.transpose(weights_normed))
        logits_pos = tf.boolean_mask(logits_glob, label_mask_pos_glob)
        logits_neg = tf.boolean_mask(logits_glob, label_mask_neg_glob)


        if scale == 'auto':
            # Automatic learned scale
            scale = tf.log(tf.exp(0.0) + tf.exp(_scale))
        else:
            # Assigned scale value
            assert type(scale) == float
            scale = tf.constant(scale)

        # Losses
        logits_pos = tf.reshape(logits_pos, [batch_size, -1])
        logits_neg = tf.reshape(logits_neg, [batch_size, -1])

        logits_pos = logits_pos * scale
        logits_neg = logits_neg * scale
        logits_neg = tf.reduce_logsumexp(logits_neg, axis=1)[:,None]

        loss_ = tf.nn.softplus(m + logits_neg - logits_pos)
        loss = tf.reduce_mean(loss_, name='am_softmax')

        # Update centers
        if not weights in tf.trainable_variables():
            if multi_lr:
                alpha = alpha * learning_rate
            if weights_target is None:
                print('Imprinting target...')
                weights_target = prelogits_normed
                label_target = label
            weights_batch = tf.gather(weights, label_target)
            diff_centers = weights_batch - weights_target
            unique_label, unique_idx, unique_count = tf.unique_with_counts(label_target)
            appear_times = tf.gather(unique_count, unique_idx)
            appear_times = tf.reshape(appear_times, [-1, 1])
            diff_centers = diff_centers / tf.cast(appear_times, tf.float32)
            diffTrue_centers = alpha * diff_centers
            centers_update_op = tf.scatter_sub(weights, label_target, diff_centers)
            with tf.control_dependencies([centers_update_op]):
            #     weights_batch = tf.gather(weights, label)
            #     weights_new = tf.nn.l2_normalize(weights_batch, dim=1)
            #     centers_update_op = tf.scatter_update(weights, label, weights_new)
                centers_update_op = tf.assign(weights, tf.nn.l2_normalize(weights,dim=1))
            #     centers_update_op = tf.group(centers_update_op)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, centers_update_op)

        # Analysis
        tfwatcher.insert('scale', scale)

    return loss

def matrix_normalize(w, iterations=5):
    ''' Sinkhorn-Knopp algorithm for matrix normalization. Sinkhorn's theorem.'''
    for i in range(iterations):
        w = w / tf.reduce_sum(w, axis=1, keep_dims=True)
        w = w / tf.reduce_sum(w, axis=0, keep_dims=True)
    return w

def group_loss(prelogits, weights, label, num_classes, global_step, weight_decay, 
                scale=16.0, m=1.0, alpha=None, reuse=None):
    n_groups = weights.shape[1].value
    # weights = tf.nn.softmax(weights, axis=0)
    weights = matrix_normalize(tf.exp(weights))
    prelogits_splits =  tf.split(prelogits, n_groups, axis=1)
    weights_splits = tf.split(weights, n_groups, axis=1)
    from functools import partial
    loss = partial(am_softmax, num_classes=num_classes, global_step=global_step, 
            weight_decay=weight_decay, scale=scale, m=m, alpha=alpha, reduce_mean=False, reuse=reuse)

    print('{} groups'.format(n_groups))

    loss_all = 0.
    for i in range(n_groups):
        prelogits_splits[i], 
        loss_all += tf.reduce_sum(weights_splits[i] * loss(prelogits, label, scope='loss_{}'.format(i)))
        tfwatcher.insert('w{}'.format(i+1), tf.reduce_sum(weights_splits[i]))

    if False:
        weight_reg = tf.reduce_sum(weights, axis=0)
        weight_reg = weight_reg / tf.reduce_sum(weight_reg)
        weight_reg = tf.reduce_mean(-tf.log(weight_reg))
        loss_all += weight_reg
        tfwatcher.insert('wreg', weight_reg)
 
    weight_std = tf.reduce_mean(tf.nn.moments(weights, axes=[1])[1])   
    tfwatcher.insert('wstd', weight_std)
    tf.summary.image('weight', weights[None, :32, :, None])

    return loss_all

def euc_loss(prelogits, label, num_classes, global_step, weight_decay,
                    m=1.0, alpha=0.5, reuse=None):
    num_features = prelogits.shape[1].value
    batch_size = tf.shape(prelogits)[0]
    with tf.variable_scope('EucLoss', reuse=reuse):
        weights = tf.get_variable('weights', shape=(num_classes, num_features),
                # initializer=slim.xavier_initializer(),
                # initializer=tf.truncated_normal_initializer(stddev=0.05),
                initializer=tf.constant_initializer(0),
                trainable=True,
                dtype=tf.float32)


        if False:
            prelogits_ = tf.reshape(prelogits, [-1, 1, num_features])
            weights_ = tf.reshape(weights, [1, -1, num_features])
            dist = np.square(prelogits_ - weights_)
            dist = tf.transpose(dist, [0, 2, 1])
            dist = tf.reshape(dist, [-1, num_classes])
            logits = - 0.5 * dist
        

            label_ = tf.tile(tf.reshape(label, [-1,1]), [1, num_features])
            label_ = tf.reshape(label_, [-1])

            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
                labels=label_, logits=logits), name='loss')

        else:
            weights_ = weights # batch_norm(weights)
            # Label and logits between batch and examplars
            label_mat_glob = tf.one_hot(label, num_classes, dtype=tf.float32)
            label_mask_pos_glob = tf.cast(label_mat_glob, tf.bool)
            label_mask_neg_glob = tf.logical_not(label_mask_pos_glob)

            dist_glob = euclidean_distance(prelogits, tf.transpose(weights_))
            dist_pos = tf.boolean_mask(dist_glob, label_mask_pos_glob)
            dist_neg = tf.boolean_mask(dist_glob, label_mask_neg_glob)

            # Losses
            dist_pos = tf.log(tf.reshape(dist_pos, [batch_size, -1]) + 1e-6)
            dist_neg = tf.log(tf.reshape(dist_neg, [batch_size, -1]) + 1e-6)
            dist_neg = -tf.reduce_logsumexp(-dist_neg, axis=1)[:,None]

            #loss_pos = tf.reduce_mean(dist_pos)
            #loss_neg = tf.reduce_mean(tf.nn.relu(m - dist_neg))

            loss = tf.reduce_logsumexp(tf.nn.softplus(m + dist_pos - dist_neg))

            tfwatcher.insert('mean_mu', tf.reduce_mean(tf.norm(weights, axis=1)))
            tfwatcher.insert('mean_feat', tf.reduce_mean(tf.norm(prelogits, axis=1)))


        # Update centers
        if not weights in tf.trainable_variables():
            weights_target = prelogits
            label_target = label
            weights_batch = tf.gather(weights, label_target)
            diff_centers = weights_batch - weights_target
            unique_label, unique_idx, unique_count = tf.unique_with_counts(label_target)
            appear_times = tf.gather(unique_count, unique_idx)
            appear_times = tf.reshape(appear_times, [-1, 1])
            diff_centers = diff_centers / tf.cast(appear_times, tf.float32)
            diff_centers = alpha * diff_centers
            centers_update_op = tf.scatter_sub(weights, label_target, diff_centers)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, centers_update_op)


    return loss

def split_softmax(prelogits, label, num_classes, 
                global_step, weight_decay, gamma=16.0, m=1.0, reuse=None):
    nrof_features = prelogits.shape[1].value
    batch_size = tf.shape(prelogits)[0]
    with tf.variable_scope('SplitSoftmax', reuse=reuse):
        weights = tf.get_variable('weights', shape=(num_classes, nrof_features),
                regularizer=slim.l2_regularizer(weight_decay),
                initializer=slim.xavier_initializer(),
                # initializer=tf.truncated_normal_initializer(stddev=0.5),
                # initializer=tf.constant_initializer(0),
                trainable=True,
                dtype=tf.float32)
        alpha = tf.get_variable('alpha', shape=(),
                regularizer=slim.l2_regularizer(1e-4),
                initializer=tf.constant_initializer(1.00),
                trainable=True,
                dtype=tf.float32)
        beta = tf.get_variable('beta', shape=(),
                # regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(0.0),
                trainable=True,
                dtype=tf.float32)

        # Normalizing the vecotors
        # weights_normed = weights
        weights_normed = tf.nn.l2_normalize(weights, dim=1)
        # prelogits_normed = prelogits
        prelogits_normed = tf.nn.l2_normalize(prelogits, dim=1)
        # norm_ = tf.norm(prelogits_normed, axis=1)
        # tfwatcher.insert('pnorm', tf.reduce_mean(norm_))
        

        coef = -1.0
        # Label and logits between batch and examplars
        label_mat_glob = tf.one_hot(label, num_classes, dtype=tf.float32)
        label_mask_pos_glob = tf.cast(label_mat_glob, tf.bool)
        label_mask_neg_glob = tf.logical_not(label_mask_pos_glob)
        # label_exp_batch = tf.expand_dims(label, 1)
        # label_exp_glob = tf.expand_dims(label_history, 1)
        # label_mat_glob = tf.equal(label_exp_batch, tf.transpose(label_exp_glob))
        # label_mask_pos_glob = tf.cast(label_mat_glob, tf.bool)
        # label_mask_neg_glob = tf.logical_not(label_mat_glob)

        dist_mat_glob = euclidean_distance(prelogits_normed, tf.transpose(weights_normed), False)
        # dist_mat_glob = tf.matmul(prelogits_normed, tf.transpose(weights_normed))
        dist_pos_glob = tf.boolean_mask(dist_mat_glob, label_mask_pos_glob)
        dist_neg_glob = tf.boolean_mask(dist_mat_glob, label_mask_neg_glob)

        logits_glob = coef * dist_mat_glob
        logits_pos_glob = tf.boolean_mask(logits_glob, label_mask_pos_glob)
        logits_neg_glob = tf.boolean_mask(logits_glob, label_mask_neg_glob)


        # Label and logits within batch
        label_exp_batch = tf.expand_dims(label, 1)
        label_mat_batch = tf.equal(label_exp_batch, tf.transpose(label_exp_batch))
        label_mask_pos_batch = tf.cast(label_mat_batch, tf.bool)
        label_mask_neg_batch = tf.logical_not(label_mask_pos_batch)
        mask_non_diag = tf.logical_not(tf.cast(tf.eye(batch_size), tf.bool))
        label_mask_pos_batch = tf.logical_and(label_mask_pos_batch, mask_non_diag)

        dist_mat_batch = euclidean_distance(prelogits_normed, tf.transpose(prelogits_normed), False)
        # dist_mat_batch = tf.matmul(prelogits_normed, tf.transpose(prelogits_normed))
        dist_pos_batch = tf.boolean_mask(dist_mat_batch, label_mask_pos_batch)
        dist_neg_batch = tf.boolean_mask(dist_mat_batch, label_mask_neg_batch)

        logits_batch =  coef * dist_mat_batch
        logits_pos_batch = tf.boolean_mask(logits_batch, label_mask_pos_batch)
        logits_neg_batch = tf.boolean_mask(logits_batch, label_mask_neg_batch)
        
        dist_pos = dist_pos_batch
        dist_neg = dist_neg_batch
        logits_pos = logits_pos_batch
        logits_neg = logits_neg_batch


        if gamma == 'auto':
            # gamma = tf.nn.softplus(alpha)
            gamma = tf.log(tf.exp(1.0) + tf.exp(alpha))
        elif type(gamma) == tuple:
            t_min, decay = gamma
            epsilon = 1.0
            t = tf.maximum(t_min, 1.0/(epsilon + decay*tf.cast(global_step, tf.float32)))
            gamma = 1.0 / t
        else:
            assert type(gamma) == float
            gamma = tf.constant(gamma)

        # Losses

        t_pos = (beta)
        t_neg = (beta)

        logits_pos = tf.reshape(logits_pos, [batch_size, -1])
        logits_neg = tf.reshape(logits_neg, [batch_size, -1])

        logits_pos = tf.stop_gradient(logits_pos) * gamma + beta
        logits_neg = tf.stop_gradient(logits_neg) * gamma + beta
        loss_gb = tf.reduce_mean(tf.nn.softplus(-logits_pos)) + tf.reduce_mean(tf.nn.softplus(logits_neg))

        g,b = tf.stop_gradient(gamma), tf.stop_gradient(beta)
        loss_pos = tf.reduce_mean(g*dist_pos)
        loss_neg = tf.nn.softplus(tf.reduce_logsumexp(b - g * dist_neg))
        # _logits_neg = tf.reduce_logsumexp(_logits_neg, axis=1)[:,None]
        
        # _logits_neg = -tf.reduce_max(-_logits_neg, axis=1)[:,None]
        
        tfwatcher.insert('neg', tf.reduce_mean(loss_neg))
        tfwatcher.insert('pos', tf.reduce_mean(loss_pos))
        tfwatcher.insert('lneg', tf.reduce_mean(logits_neg))
        tfwatcher.insert('lpos', tf.reduce_mean(logits_pos))
        

        #num_violate = tf.reduce_sum(tf.cast(tf.greater(m + _logits_neg - _logits_pos, 0.), tf.float32), axis=1, keep_dims=True)
        #loss =  tf.reduce_sum(tf.nn.relu(m + _logits_neg - _logits_pos), axis=1, keep_dims=True) / (num_violate + 1e-8)
        #tfwatcher.insert('nv', tf.reduce_mean(num_violate))
        
        # loss = tf.nn.softplus(m + _logits_neg - _logits_pos)

        loss = tf.reduce_mean(loss_gb + loss_pos + loss_neg, name='split_loss')


        # Update centers
        if not weights in tf.trainable_variables():
            weights_batch = tf.gather(weights, label)
            diff_centers = weights_batch - prelogits_normed
            unique_label, unique_idx, unique_count = tf.unique_with_counts(label)
            appear_times = tf.gather(unique_count, unique_idx)
            appear_times = tf.reshape(appear_times, [-1, 1])
            diff_centers = diff_centers / tf.cast((1 + appear_times), tf.float32)
            diff_centers = 0.5 * diff_centers
            centers_update_op = tf.scatter_sub(weights, label, diff_centers)
            with tf.control_dependencies([centers_update_op]):
                centers_update_op = tf.assign(weights, tf.nn.l2_normalize(weights,dim=1))
            # centers_decay_op = tf.assign_sub(weights, 2*weight_decay*weights)# weight decay
            centers_update_op = tf.group(centers_update_op)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, centers_update_op)

        # Analysis
        tf.summary.scalar('gamma', gamma)
        # tf.summary.scalar('alpha', alpha)
        tf.summary.scalar('beta', beta)
        tfwatcher.insert('gamma', gamma)
        tfwatcher.insert('beta', beta)

    return loss

def centers_by_label(features, label):
    # Compute centers within batch
    unique_label, unique_idx, unique_count = tf.unique_with_counts(label)
    num_centers = tf.size(unique_label)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])
    weighted_prelogits = features / tf.cast(appear_times, tf.float32)
    centers = tf.unsorted_segment_sum(weighted_prelogits, unique_idx, num_centers)
    return centers, unique_label, unique_idx, unique_count


def pair_loss(prelogits, label, num_classes, 
                global_step, weight_decay, gamma=16.0, m=1.0, reuse=None):
    nrof_features = prelogits.shape[1].value
    batch_size = tf.shape(prelogits)[0]
    with tf.variable_scope('PairLoss', reuse=reuse):
        weights = tf.get_variable('weights', shape=(num_classes, nrof_features),
                regularizer=slim.l2_regularizer(weight_decay),
                initializer=slim.xavier_initializer(),
                # initializer=tf.truncated_normal_initializer(stddev=0.0),
                # initializer=tf.constant_initializer(0),
                trainable=True,
                dtype=tf.float32)
        alpha = tf.get_variable('alpha', shape=(),
                regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(1.00),
                trainable=True,
                dtype=tf.float32)
        beta = tf.get_variable('beta', shape=(),
                # regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(0.0),
                trainable=True,
                dtype=tf.float32)

        # Normalizing the vecotors
        weights_normed = tf.nn.l2_normalize(weights, dim=1)
        prelogits_normed = tf.nn.l2_normalize(prelogits, dim=1)
        # weights_normed = weights
        # prelogits_normed = prelogits

        prelogits_reshape = tf.reshape(prelogits_normed, [-1,2,tf.shape(prelogits_normed)[1]])
        prelogits_tmp = prelogits_reshape[:,0,:]
        prelogits_pro = prelogits_reshape[:,1,:]
    
        dist_mat_batch = -euclidean_distance(prelogits_tmp, tf.transpose(prelogits_pro), False)
        # dist_mat_batch = tf.matmul(prelogits_tmp, tf.transpose(prelogits_pro))
        
        logits_mat_batch = dist_mat_batch

        num_pairs = tf.shape(prelogits_reshape)[0]
        label_mask_pos_batch = tf.cast(tf.eye(num_pairs), tf.bool)
        label_mask_neg_batch = tf.logical_not(label_mask_pos_batch)
        dist_pos_batch = tf.boolean_mask(dist_mat_batch, label_mask_pos_batch)
        dist_neg_batch = tf.boolean_mask(dist_mat_batch, label_mask_neg_batch)

        logits_pos_batch = tf.boolean_mask(logits_mat_batch, label_mask_pos_batch)
        logits_neg_batch = tf.boolean_mask(logits_mat_batch, label_mask_neg_batch)

        logits_pos = logits_pos_batch
        logits_neg = logits_neg_batch
    
        dist_pos = dist_pos_batch
        dist_neg = dist_neg_batch


        if gamma == 'auto':
            # gamma = tf.nn.softplus(alpha)
            gamma = tf.log(tf.exp(1.0) + tf.exp(alpha))
        elif type(gamma) == tuple:
            t_min, decay = gamma
            epsilon = 1.0
            t = tf.maximum(t_min, 1.0/(epsilon + decay*tf.cast(global_step, tf.float32)))
            gamma = 1.0 / t
        else:
            assert type(gamma) == float
            gamma = tf.constant(gamma)

        hinge_loss = lambda x: tf.nn.relu(1.0 + x)
        margin_func = hinge_loss

        # Losses
        losses = []

        t_pos = (beta)
        t_neg = (beta)

        _logits_pos = tf.reshape(logits_pos, [num_pairs, -1])
        _logits_neg_1 = tf.reshape(logits_neg, [num_pairs, -1])
        _logits_neg_2 = tf.reshape(logits_neg, [-1, num_pairs])

        _logits_pos = _logits_pos * gamma
        _logits_neg_1 = tf.reduce_max(_logits_neg_1, axis=1)[:,None]
        _logits_neg_2 = tf.reduce_max(_logits_neg_2, axis=0)[:,None]
        _logits_neg = tf.maximum(_logits_neg_1, _logits_neg_2)
        # _logits_neg_1 = tf.reduce_logsumexp(gamma*_logits_neg_1, axis=1)[:,None]
        # _logits_neg_2 = tf.reduce_logsumexp(gamma*_logits_neg_2, axis=0)[:,None]

        loss_pos = tf.nn.relu(m + _logits_neg_1 - _logits_pos) * 0.5
        loss_neg = tf.nn.relu(m + _logits_neg_2 - _logits_pos) * 0.5
        loss = tf.reduce_mean(loss_pos + loss_neg)
        loss = tf.identity(loss, name='pair_loss')
        losses.extend([loss])
        tfwatcher.insert('ploss', loss)

        # Analysis
        tf.summary.scalar('gamma', gamma)
        tf.summary.scalar('alpha', alpha)
        tf.summary.scalar('beta', beta)
        tf.summary.histogram('dist_pos', dist_pos)
        tf.summary.histogram('dist_neg', dist_neg)

        tfwatcher.insert('gamma', gamma)

    return losses



def pair_loss_twin(prelogits_tmp, prelogits_pro, label_tmp, label_pro, num_classes, 
                global_step, weight_decay, gamma=16.0, m=1.0, reuse=None):
    num_features = prelogits_tmp.shape[1].value
    batch_size = tf.shape(prelogits_tmp)[0] + tf.shape(prelogits_pro)[0]
    with tf.variable_scope('PairLoss', reuse=reuse):
        alpha = tf.get_variable('alpha', shape=(),
                # regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(1.00),
                trainable=True,
                dtype=tf.float32)
        beta = tf.get_variable('beta', shape=(),
                # regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(0.0),
                trainable=True,
                dtype=tf.float32)

        # Normalizing the vecotors
        prelogits_tmp = tf.nn.l2_normalize(prelogits_tmp, dim=1)
        prelogits_pro = tf.nn.l2_normalize(prelogits_pro, dim=1)
    
        dist_mat_batch = -euclidean_distance(prelogits_tmp, tf.transpose(prelogits_pro),False)
        # dist_mat_batch = tf.matmul(prelogits_tmp, tf.transpose(prelogits_pro))
        
        logits_mat_batch = dist_mat_batch

        num_pairs = tf.shape(prelogits_tmp)[0]

        # label_tmp_batch = tf.expand_dims(label_tmp, 1)
        # label_pro_batch = tf.expand_dims(label_pro, 1)
        # label_mat_batch = tf.equal(label_tmp_batch, tf.transpose(label_pro_batch))
        # label_mask_pos_batch = tf.cast(label_mat_batch, tf.bool)
        label_mask_pos_batch = tf.cast(tf.eye(num_pairs), tf.bool)
        label_mask_neg_batch = tf.logical_not(label_mask_pos_batch)

        logits_pos = tf.boolean_mask(logits_mat_batch, label_mask_pos_batch)
        logits_neg_1 = tf.boolean_mask(logits_mat_batch, label_mask_neg_batch)
        logits_neg_2 = tf.boolean_mask(tf.transpose(logits_mat_batch), label_mask_neg_batch)

        if gamma == 'auto':
            # gamma = tf.nn.softplus(alpha)
            gamma = tf.log(tf.exp(1.0) + tf.exp(alpha))
        elif type(gamma) == tuple:
            t_min, decay = gamma
            epsilon = 1.0
            t = tf.maximum(t_min, 1.0/(epsilon + decay*tf.cast(global_step, tf.float32)))
            gamma = 1.0 / t
        else:
            assert type(gamma) == float
            gamma = tf.constant(gamma)

        # Losses
        losses = []

        t_pos = (beta)
        t_neg = (beta)

        _logits_pos = tf.reshape(logits_pos, [num_pairs, -1])
        _logits_neg_1 = tf.reshape(logits_neg_1, [num_pairs, -1])
        _logits_neg_2 = tf.reshape(logits_neg_2, [num_pairs, -1])
        
        
        _logits_neg_1 = tf.reduce_max(_logits_neg_1, axis=1)[:,None]
        _logits_neg_2 = tf.reduce_max(_logits_neg_2, axis=1)[:,None]
        _logits_neg = tf.maximum(_logits_neg_1, _logits_neg_2)
        # _logits_neg = tf.concat([_logits_neg_1, _logits_neg_2], axis=1)
        # _logits_neg = tf.reduce_logsumexp(_logits_neg, axis=1)[:,None]


        num_violate = tf.reduce_sum(tf.cast(tf.greater(m + _logits_neg - _logits_pos, 0.), tf.float32), axis=1, keep_dims=True)

        loss_1 = tf.reduce_sum(tf.nn.relu(m + _logits_neg - _logits_pos), axis=1, keep_dims=True) * 0.5 # / (num_violate + 1e-8)
        loss_2 = tf.reduce_sum(tf.nn.relu(m + _logits_neg - _logits_pos), axis=1, keep_dims=True) * 0.5 # / (num_violate + 1e-8)
        loss = tf.reduce_mean(loss_1 + loss_2)
        loss = tf.identity(loss, name='pair_loss')
        losses.extend([loss])

        # Analysis
        tf.summary.scalar('gamma', gamma)
        tf.summary.scalar('alpha', alpha)
        tf.summary.scalar('beta', beta)
        tf.summary.histogram('dist_pos', _logits_pos)
        tf.summary.histogram('dist_neg', _logits_neg)

        tfwatcher.insert("gamma", gamma)

    return losses

def l2centers(features, label, centers, coef):
    centers_batch = tf.gather(centers, label)
    loss = tf.reduce_mean(coef * tf.reduce_sum(tf.square(features - centers_batch), axis=1), name='l2centers')
    
    return loss

def pair_regression(features, targets, coef):
    features = tf.nn.l2_normalize(features, dim=1)
    targets = tf.nn.l2_normalize(targets, dim=1)

    loss = coef * tf.reduce_mean(tf.reduce_sum(tf.square(features - targets), axis=1))
    
    return loss


def triplet_loss_func(anchor,positive,negative, alpha=0.3,normalize= True):
    '''
    Used directly as loss function
        Inputs:
                    y_true: True values of classification. (y_train)
                    y_pred: predicted values of classification.
                    alpha: Distance between positive and negative sample, arbitrarily
                           set to 0.3
        Returns:
                    Computed loss
        Function:
                    --Implements triplet loss using tensorflow commands
                    --The following function follows an implementation of Triplet-Loss 
                      where the loss is applied to the network in the compile statement 
                      as usual.
    '''
    if(normalize == True):
        pos = tf.nn.l2_normalize(pos,dim=1)
        anchor = tf.nn.l2_normalize(anchor, dim=1)
        neg = tf.nn.l2_normalize(neg,dim=1)

    positive_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), -1)
    negative_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)), -1)

    loss_1 = tf.add(tf.subtract(positive_dist, negative_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(loss_1, 0.0))

    return loss

def reduce_mean_nonzero(tensor, axis):
    return tf.reduce_sum(tensor, axis=axis) / (tf.reduce_sum(tf.cast(tf.greater(tensor, 0), tf.float32), axis=axis) + 1e-8)

def triplet_loss_divyansh(pos,anchor,neg,margin,normalize=True):

    batch_size = pos.shape[0]
    if(normalize==True):
        pos = tf.nn.l2_normalize(pos, dim=1)
        anchor = tf.nn.l2_normalize(anchor, dim=1)
        neg = tf.nn.l2_normalize(neg, dim=1)

    dist_pos = tf.reduce_sum(tf.square(pos-anchor),axis=1)
    dist_neg = tf.reduce_sum(tf.square(anchor-neg),axis=1)

    # dist_pos = tf.reshape(dist_pos, [batch_size,-1])
    # dist_neg = tf.reshape(dist_neg, [batch_size, -1])

    #dist_neg = -tf.reduce_max(-dist_neg, axis=1, keep_dims=True)
    loss = tf.nn.relu(dist_pos - dist_neg + margin)
    loss = tf.reduce_mean(loss, name='TripletLoss')
    return loss

def fixed_anchor(pos_embeddings, anchor, neg_embeddings,normalize=True, margin=1.0):
    # pos_embeddings = tf.boolean_mask(embeddings, tf.equal(labels, 0))
    # neg_embeddings = tf.boolean_mask(embeddings, tf.equal(labels, 1))
    # Get per pair distances
    # anchor = tf.constant([1.0] + [0.0]*(embeddings.shape[1].value-1), name='anchor')

    if(normalize==True):
        pos_embeddings = tf.nn.l2_normalize(pos_embeddings, dim=-1)
        anchor = tf.nn.l2_normalize(anchor, dim=-1)
        neg_embeddings = tf.nn.l2_normalize(neg_embeddings, dim=-1)

    dist_pos = euclidean_distance(pos_embeddings, tf.transpose(anchor), False)
    dist_neg = euclidean_distance(anchor, tf.transpose(neg_embeddings), False)

    dist = dist_pos - dist_neg # [Np, Np - 1, Nn]
    loss = tf.nn.relu(dist + margin)
    loss = reduce_mean_nonzero(loss, axis=1)
    loss = tf.reduce_mean(loss)
    return loss


def triplet_loss(labels, embeddings, margin, normalize=False):

    with tf.name_scope('TripletLoss'):
    
        if normalize:
            embeddings = tf.nn.l2_normalize(embeddings, dim=1)

        batch_size = tf.shape(embeddings)[0]
        num_features = embeddings.shape[1].value


        diag_mask = tf.eye(batch_size, dtype=tf.bool)
        non_diag_mask = tf.logical_not(diag_mask)    

        dist_mat = euclidean_distance(embeddings, tf.transpose(embeddings), False)
        

        label_mat = tf.equal(labels[:,None], labels[None,:])
        label_mask_pos = tf.logical_and(non_diag_mask, label_mat)
        label_mask_neg = tf.logical_and(non_diag_mask, tf.logical_not(label_mat))

        
        dist_pos = tf.boolean_mask(dist_mat, label_mask_pos) 
        dist_neg = tf.boolean_mask(dist_mat, label_mask_neg)
        
        dist_pos = tf.reshape(dist_pos, [batch_size, -1])
        dist_neg = tf.reshape(dist_neg, [batch_size, -1])


        # Hard Negative Mining
        dist_neg = -tf.reduce_max(-dist_neg, axis=1, keep_dims=True)
        
        loss = tf.nn.relu(dist_pos - dist_neg + margin)
        loss = tf.reduce_mean(loss, name='TripletLoss')
    
    return loss

def average_nonneg(losses, axis=None):
    
    valid = tf.cast(tf.greater(losses, 0.0), tf.float32)
    valid_count = tf.reduce_sum(valid, axis=axis)
    loss = tf.reduce_sum(losses, axis=axis) / (valid_count + 1e-8)

    return loss



def uncertain_pair_loss(mu, log_sigma_sq, coef=1.0):
    with tf.name_scope('UncertainPairLoss'):
        batch_size = tf.shape(mu)[0]
        num_features = mu.shape[1].value
 
        mu = tf.reshape(mu, [-1, 2, num_features])
        sigma_sq = tf.exp(tf.reshape(log_sigma_sq, [-1, 2, num_features]))
        
        mu1, mu2 = mu[:,0], mu[:,1]
        sigma_sq1, sigma_sq2 = sigma_sq[:,0], sigma_sq[:,1]

        mu_diff = mu1 - mu2
        sigma_sq_sum = sigma_sq1 + sigma_sq2

        eps = 1e-8
        loss = tf.square(mu_diff) / (eps + sigma_sq_sum) + tf.log(sigma_sq_sum)
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    
        return loss 

def masked_reduce_mean(tensor, mask, axis, eps=1e-8):
    mask = tf.cast(mask, tf.float32)
    num_valid = tf.reduce_sum(mask, axis=axis)
    return tf.reduce_sum(tensor * mask, axis=axis)  / (num_valid + eps)

def triplet_avghard_loss(labels, embeddings, margin=1., normalize=False):

    with tf.name_scope('AvgHardTripletLoss'):
    
        if normalize:
            embeddings = tf.nn.l2_normalize(embeddings, dim=1)

        batch_size = tf.shape(embeddings)[0]
        num_features = embeddings.shape[1].value

        dist_mat = euclidean_distance(embeddings, tf.transpose(embeddings), False)
        
        diag_mask = tf.eye(batch_size, dtype=tf.bool)
        non_diag_mask = tf.logical_not(diag_mask)
        label_mat = tf.equal(labels[:,None], labels[None,:])
        label_mask_pos = tf.logical_and(non_diag_mask, label_mat)
        label_mask_neg = tf.logical_not(label_mat)


        # Followings are different from hard negative mining

        dist_tensor_neg = tf.tile(dist_mat, [1, batch_size])
        dist_tensor_neg = tf.reshape(dist_tensor_neg, [batch_size, batch_size, batch_size])

        label_tensor_neg= tf.tile(label_mask_neg, [1, batch_size])
        label_tensor_neg = tf.reshape(label_tensor_neg, [batch_size, batch_size, batch_size])

        loss = tf.nn.relu(dist_mat[:,:,None] - dist_tensor_neg + margin)

        # Mask the third dimension to pick the negative samples
        mask = tf.logical_and(label_tensor_neg, tf.greater(loss, 0.))
        loss = masked_reduce_mean(loss, mask, axis=2)

        # Mask the first two dimension to only keep positive pairs
        loss = tf.boolean_mask(loss, label_mask_pos)
        loss = tf.reduce_mean(loss)

        return loss


def uncertain_triplet_loss(labels, embeddings, log_sigma_sq, margin=100., coef=1.0, normalize=False):

    with tf.name_scope('TripletLoss'):
    
        if normalize:
            embeddings = tf.nn.l2_normalize(embeddings, dim=1)

        batch_size = tf.shape(embeddings)[0]
        num_features = embeddings.shape[1].value


        diag_mask = tf.eye(batch_size, dtype=tf.bool)
        non_diag_mask = tf.logical_not(diag_mask)

        sigma_sq = tf.exp(log_sigma_sq)
        dist_mat = uncertain_distance(embeddings, embeddings, sigma_sq, sigma_sq)
        
        # dist_mat_ = uncertain_distance(embeddings, tf.transpose(embeddings), tf.stop_gradient(sigma_sq), tf.stop_gradient(sigma_sq))
        
        label_mat = tf.equal(labels[:,None], labels[None,:])
        label_mask_pos = tf.logical_and(non_diag_mask, label_mat)
        label_mask_neg = tf.logical_and(non_diag_mask, tf.logical_not(label_mat))

        if False:
            dist_mat_tile = tf.tile(tf.reshape(dist_mat, [batch_size, 1, -1]), [1, batch_size ,1])
            dist_mat_tile = tf.reshape(dist_mat_tile, [-1, batch_size])

            label_mat_tile = tf.tile(tf.reshape(label_mat, [batch_size, 1, -1]), [1, batch_size, 1])
            label_mat_tile = tf.reshape(label_mat_tile, [-1, batch_size])
            

            dist_flatten = tf.reshape(dist_mat, [-1, 1])
            label_flatten = tf.reshape(label_mask_pos, [-1])

            loss = dist_flatten - dist_mat_tile + margin  

            valid = tf.cast( tf.logical_and(tf.logical_not(label_mat_tile), tf.greater(loss, 0.0)), tf.float32)          
            loss = tf.nn.relu(loss) 

            # Set rows of impostor pairs to zero
            # valid = tf.cast(tf.reshape(label_flatten, [-1,1]), tf.float32) * valid

            valid = tf.boolean_mask(valid, label_flatten)
            loss = tf.boolean_mask(loss, label_flatten)

            # loss = tf.reshape(loss, [batch_size, -1])
            # valid = tf.reshape(valid, [batch_size, -1])

            valid_count = tf.reduce_sum(valid, axis=1) + 1e-8
            
            loss = tf.reduce_sum(loss * valid, axis=1) / valid_count

            loss = tf.reduce_mean(loss)

        else:   
            dist_pos = tf.boolean_mask(dist_mat, label_mask_pos) 
            dist_neg = tf.boolean_mask(dist_mat, label_mask_neg)
        
            dist_pos = tf.reshape(dist_pos, [-1,1])
            dist_neg = tf.reshape(dist_neg, [1,-1])
            # return tf.reduce_mean(dist_pos)
            
            if True:
                loss = dist_pos - dist_neg + margin
                valid = tf.cast(tf.greater(loss, 0.0), tf.float32)
                valid_count = tf.reduce_sum(valid, axis=1) + 1e-8
                loss = tf.reduce_sum(tf.nn.relu(loss) * valid, axis=1) / valid_count
                loss = tf.reduce_mean(dist_pos)
            elif False:
                loss_pos = tf.reduce_mean(dist_pos)
                loss_neg = tf.stop_gradient(dist_pos) - dist_neg + margin
                valid = tf.cast(tf.greater(loss_neg, 0.0), tf.float32)
                valid_count = tf.reduce_sum(valid, axis=1) + 1e-8
                loss_neg = tf.reduce_mean(tf.reduce_sum(valid* loss_neg, axis=1) / valid_count)
                loss = loss_pos + loss_neg
            else:
                # margin = tf.stop_gradient(tf.reduce_mean(dist_pos))
                print('Constrastive Loss: margin={}'.format(margin))
                loss_pos = tf.reduce_mean(dist_pos)
                loss_neg = tf.reduce_mean(tf.nn.relu(margin - dist_neg))
                # loss_neg = average_nonneg(tf.nn.relu(margin - dist_neg))
                loss = loss_pos + loss_neg
                tfwatcher.insert('lneg', loss_neg)

        loss = coef * loss
     
        # dist_pos = tf.boolean_mask(dist_mat, label_mask_pos) 
        # dist_neg = tf.boolean_mask(dist_mat_, label_mask_neg)
        
        # dist_pos = tf.reshape(dist_pos, [batch_size, -1])
        # dist_neg = tf.reshape(dist_neg, [batch_size, -1])


        # Hard Negative Mining
        # dist_neg = -tf.reduce_max(-dist_neg, axis=1, keep_dims=True)
        
        # loss = tf.reduce_mean(dist_pos) - tf.reduce_mean(dist_neg)
        # loss = tf.nn.relu(dist_pos - dist_neg + margin)
        # loss = tf.reduce_mean(dist_pos - dist_neg)
        # loss = tf.reduce_mean(loss, name='TripletLoss')

        tfwatcher.insert('mean_sigma', tf.reduce_mean(tf.exp(0.5*log_sigma_sq)))
    
    return loss

def contrastive_loss(labels, embeddings, margin, normalize=False):

    with tf.name_scope('ContrastiveLoss'):

        if normalize:
            embeddings = tf.nn.l2_normalize(embeddings, dim=1)

        batch_size = tf.shape(embeddings)[0]
        num_features = embeddings.shape[1].value


        diag_mask = tf.eye(batch_size, dtype=tf.bool)
        non_diag_mask = tf.logical_not(diag_mask)    

        dist_mat = euclidean_distance(embeddings, tf.transpose(embeddings), False)
        

        label_mat = tf.equal(labels[:,None], labels[None,:])
        label_mask_pos = tf.logical_and(non_diag_mask, label_mat)
        label_mask_neg = tf.logical_and(non_diag_mask, tf.logical_not(label_mat))

        
        dist_pos = tf.boolean_mask(dist_mat, label_mask_pos) 
        dist_neg = tf.boolean_mask(dist_mat, label_mask_neg)

        # Keep hards triplets
        # dist_pos = tf.reshape(dist_pos, [batch_size, -1])
        # dist_neg = tf.reshape(dist_neg, [batch_size, -1])
        # dist_neg = tf.reduce_min(dist_neg, axis=1, keep_dims=True)

        loss_pos = tf.reduce_mean(dist_pos)
        loss_neg = tf.reduce_mean(tf.nn.relu(margin - dist_neg))
        
        loss = tf.identity(loss_pos + loss_neg, name='contrastive_loss')
    
    return loss

def scaled_npair(prelogits, labels, num_classes, 
                scale='auto', scale_decay=1e-2, m=1.0, reuse=None):
    num_features = prelogits.shape[1].value
    batch_size = tf.shape(prelogits)[0]
    with tf.variable_scope('NPairLoss', reuse=reuse):
        _scale = tf.get_variable('_scale', shape=(),
                regularizer=slim.l2_regularizer(scale_decay),
                initializer=tf.constant_initializer(0.00),
                trainable=True,
                dtype=tf.float32)

        # Normalizing the vecotors
        prelogits = tf.nn.l2_normalize(prelogits, dim=1)

        # Label and logits within batch
        label_exp = tf.expand_dims(labels, 1)
        label_mat = tf.equal(label_exp, tf.transpose(label_exp))
        label_mask_pos = tf.cast(label_mat, tf.bool)
        label_mask_neg = tf.logical_not(label_mask_pos)
        mask_non_diag = tf.logical_not(tf.cast(tf.eye(batch_size), tf.bool))
        label_mask_pos = tf.logical_and(label_mask_pos, mask_non_diag)

        logits_mat = - euclidean_distance(prelogits, tf.transpose(prelogits), False)
        # logits_mat = tf.matmul(prelogits, tf.transpose(prelogits))
        logits_pos = tf.boolean_mask(logits_mat, label_mask_pos)
        logits_neg = tf.boolean_mask(logits_mat, label_mask_neg)

        if scale == 'auto':
            scale = tf.nn.softplus(_scale)
        else:
            assert type(scale) == float
            scale = tf.constant(scale)

        # Losses
        logits_pos = tf.reshape(logits_pos, [batch_size, -1])
        logits_neg = tf.reshape(logits_neg, [batch_size, -1])

        logits_pos = logits_pos * scale
        logits_neg = logits_neg * scale
        logits_neg = tf.reduce_logsumexp(logits_neg, axis=1)[:,None]

        loss_ = tf.nn.softplus(m + logits_neg - logits_pos)
        loss = tf.reduce_mean(loss_, name='npair')

        # Analysis
        tf.summary.scalar('scale', scale)
        tfwatcher.insert("scale", scale)

    return loss


def conditional_loss(z_mean, z_log_sigma_sq, labels, num_classes, global_step,  
        coef, alpha, weight_decay=5e-4,  reuse=None):
    num_features = z_mean.shape[1].value
    batch_size = tf.shape(z_mean)[0]
    with tf.variable_scope('ConditionalLoss', reuse=reuse):
        weights_ = tf.get_variable('weights', shape=(num_classes, num_features),
                regularizer=slim.l2_regularizer(weight_decay),
                initializer=slim.xavier_initializer(uniform=False),
                # initializer=tf.truncated_normal_initializer(stddev=0.0),
                # initializer=tf.constant_initializer(0),
                trainable=True,
                dtype=tf.float32)
        _scale = tf.get_variable('_scale', shape=(),
                regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(1.00),
                trainable=True,
                dtype=tf.float32)
        scale = tf.log(tf.exp(1.0) + tf.exp(_scale))
        # weights = tf.nn.l2_normalize(weights_, dim=1)
        # weights = batch_norm(tf.identity(weights))
        bn_params = {
            'decay': 0.995,
            'epsilon': 1e-8,
            'center': True,
            'scale': True,
            'trainable': True,
            'is_training': True,
            'param_initializers': {'gamma': tf.constant_initializer(0.01)},
        }
        with slim.arg_scope([slim.fully_connected], 
                normalizer_fn=None):
            pass
            weights = slim.fully_connected(weights_, num_features, 
                biases_initializer = None,
                activation_fn=None, scope='fc1')
            # weights = slim.fully_connected(weights, num_features, activation_fn=None, 
            #     normalizer_fn=None, scope='fc2')
        weights = tf.nn.l2_normalize(weights_, dim=1)

        # logits = tf.matmul(prelogits, tf.transpose(weights)) + biases
        logits = - scale * euclidean_distance(z_mean, tf.transpose(weights))

        loss_cls =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
            labels=labels, logits=logits), name='loss')


        z_inv_sigma_sq = tf.exp(-z_log_sigma_sq)

        weights_batch = tf.gather(weights, labels)
        neg_log_likelihood = 0.5 * tf.square(z_mean - weights_batch) * z_inv_sigma_sq + 0.5 * z_log_sigma_sq
        loss_nll = tf.reduce_sum(neg_log_likelihood, axis=1)
        loss_nll = tf.reduce_mean(loss_nll, name='conditional_loss')
        loss = loss_cls + coef * loss_nll

        # Update centers
        if not weights_ in tf.trainable_variables():

            labels_unique, unique_idx = tf.unique(labels)
            xc = z_mean * z_inv_sigma_sq
            xc_unique = tf.unsorted_segment_sum(xc, unique_idx, tf.shape(labels_unique)[0])
            c_unique = tf.unsorted_segment_sum(z_inv_sigma_sq, unique_idx, tf.shape(labels_unique)[0])
            xc_normed = xc_unique / c_unique

            labels_target = labels
            weights_target = z_mean
            confidence_target = z_inv_sigma_sq

            weights_batch = tf.gather(weights, labels_target)
            diff_centers = weights_batch - weights_target
            diff_centers = alpha * diff_centers * confidence_target
            centers_update_op = tf.scatter_sub(weights, labels_target, diff_centers)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, centers_update_op)

    tfwatcher.insert('mean_sigma', tf.reduce_mean(tf.exp(z_log_sigma_sq)))
    tfwatcher.insert('nll', loss_nll)
    tfwatcher.insert('scale', scale)

    return loss

def gaussian_log_likelihood(mu, log_sigma_sq, z):
    sigma_sq = tf.exp(log_sigma_sq)
    log_likelihood = log_sigma_sq + tf.square(mu - z) / sigma_sq
    log_likelihood = 0.5 * tf.reduce_sum(log_likelihood, axis=1)
    return log_likelihood

def gaussian_kl_divergence(mu1, log_sigma_sq1, mu2, log_sigma_sq2):
    sigma_sq1, sigma_sq2 = tf.exp(log_sigma_sq1), tf.exp(log_sigma_sq2)
    divergence = log_sigma_sq2 - log_sigma_sq1 + (sigma_sq1 + tf.square(mu2 - mu1)) / sigma_sq2 - 1.0
    divergence = 0.5 * tf.reduce_sum(divergence, axis=1)
    return divergence

def pair_ml_similarity(mu1, log_sigma_sq1, mu2, log_sigma_sq2):
    sigma_sq1, sigma_sq2 = tf.exp(log_sigma_sq1), tf.exp(log_sigma_sq2)
    sigma_sq_sum = sigma_sq1 + sigma_sq2
    dist = tf.log(sigma_sq_sum) + tf.square(mu2 - mu1) / sigma_sq_sum
    dist = tf.reduce_sum(dist, axis=1)
    return dist

def select_classes(weights, labels, background_size):
    
    num_classes = weights.shape[0].value

    indices = tf.random.shuffle(tf.range(num_classes))[:background_size]

    assert labels.shape.ndims == 1
    indices = tf.concat([indices, labels], axis=0)

    indices, unique_idx = tf.unique(indices)

    weights = tf.gather(weights, indices)
    labels = tf.gather(unique_idx, labels)

    return weights, labels

def class_divergence(z_mu, z_log_sigma_sq, labels, num_classes, global_step,  
        weight_decay, coef, alpha=0.1, margin=100., reuse=None):
    num_features = z_mu.shape[1].value
    batch_size = tf.shape(z_mu)[0]
    with tf.variable_scope('DivergenceLoss', reuse=reuse):
        class_mu = tf.get_variable('class_mu', shape=(num_classes, 16),
                regularizer=slim.l2_regularizer(0.0),
                # initializer=slim.xavier_initializer(),
                initializer=tf.random_normal_initializer(stddev=0.01),
                # initializer=tf.constant_initializer(0),
                trainable=True,
                dtype=tf.float32)
        class_log_sigma_sq = tf.get_variable('class_log_sigma_sq', shape=(num_classes, num_features),
                regularizer=slim.l2_regularizer(0.0),
                # initializer=slim.xavier_initializer(),
                # initializer=tf.truncated_normal_initializer(stddev=0.0),
                initializer=tf.constant_initializer(-10),
                trainable=True,
                dtype=tf.float32)

        class_mu = slim.fully_connected(class_mu, num_features, activation_fn=tf.nn.relu, scope='fc1')
        class_mu = slim.fully_connected(class_mu, num_features, activation_fn=None, scope='fc2')

        tf.add_to_collection('class_mu', class_mu)
        tf.add_to_collection('class_log_sigma_sq', class_log_sigma_sq)

        # class_mu = tf.nn.l2_normalize(class_mu, axis=1)
        # class_mu = batch_norm(class_mu, name='normed_mu')



        if True:

            # class_mu = batch_norm(tf.identity(class_mu))

            batch_mu = tf.gather(class_mu, labels)
            batch_log_sigma_sq= tf.gather(class_log_sigma_sq, labels)

            # divergence = gaussian_kl_divergence(batch_mu, batch_log_sigma_sq, z_mu, z_log_sigma_sq)
            divergence = pair_ml_similarity(batch_mu, batch_log_sigma_sq, z_mu, z_log_sigma_sq)
            # divergence = gaussian_log_likelihood(z_mu, z_log_sigma_sq, batch_mu)
            loss = tf.reduce_mean(divergence, name='divergence_loss')

            tfwatcher.insert('mean_dist', tf.reduce_mean(tf.abs(z_mu-batch_mu)))


        elif True:
            dist_mat = uncertain_distance(z_mu, class_mu, tf.exp(z_log_sigma_sq), tf.exp(class_log_sigma_sq), True)
            # dist_mat = euclidean_distance(z_mu, tf.transpose(class_mu))

            label_mat = tf.one_hot(labels, num_classes, dtype=tf.float32)
            label_mask_pos = tf.cast(label_mat, tf.bool)
            label_mask_neg = tf.logical_not(label_mask_pos)

            dist_pos = tf.boolean_mask(dist_mat, label_mask_pos)
            dist_neg = tf.boolean_mask(dist_mat, label_mask_neg)

            dist_pos = tf.reshape(dist_pos, [batch_size, 1])
            dist_neg = tf.reshape(dist_neg, [1, -1])

            valid = tf.cast(tf.greater(dist_pos - dist_neg + margin, 0.), tf.float32)
            valid_count = tf.reduce_sum(valid, axis=1) + 1e-8
            valid_count = tf.stop_gradient(valid_count)

            loss = tf.reduce_sum(tf.nn.relu(dist_pos - dist_neg + margin), axis=1) / valid_count
            # loss = tf.nn.softplus(dist_pos + tf.reduce_logsumexp(-dist_neg, axis=1) + margin)
            loss = tf.reduce_mean(loss)

        loss = coef * loss

        tfwatcher.insert('class_sigma', tf.reduce_mean(tf.exp(0.5*class_log_sigma_sq)))
        tfwatcher.insert('mean_sigma', tf.reduce_mean(tf.exp(0.5*z_log_sigma_sq)))



        if False and not class_mu in tf.trainable_variables():

            z_inv_sigma_sq = tf.exp(-z_log_sigma_sq)

            labels_unique, unique_idx, unique_count = tf.unique_with_counts(labels)
            xc = z_mu * z_inv_sigma_sq
            xc_unique = tf.unsorted_segment_sum(xc, unique_idx, tf.shape(labels_unique)[0])
            c_unique = tf.unsorted_segment_sum(z_inv_sigma_sq, unique_idx, tf.shape(labels_unique)[0])
            xc_normed = xc_unique / c_unique

            
            # tfwatcher.insert('old_c', tf.reduce_mean(z_inv_sigma_sq))
            # tfwatcher.insert('new_c', tf.reduce_mean(c_unique))
            # tfwatcher.insert('labels_unique', labels_unique)

            appear_times = tf.cast(tf.gather(unique_count, unique_idx), tf.float32)
            appear_times = tf.reshape(appear_times, [-1, 1])
            

            if True:

                # Confidence Decay 
                c_decay = 1.0
                decay_op = tf.assign(class_log_sigma_sq, class_log_sigma_sq - tf.log(c_decay))
                
                with tf.control_dependencies([decay_op]):

                    # Incremental Updating
                    mu_batch = xc_normed
                    c_batch = c_unique

                    mu_old = tf.gather(class_mu, labels_unique)
                    log_sigma_sq_old = tf.gather(class_log_sigma_sq, labels_unique)
                    c_old = tf.exp(-log_sigma_sq_old)

                    c_new = c_old + c_batch
                    mu_new = (mu_old * c_old + mu_batch * c_batch) / c_new

                    # Update variables
                    labels_target = labels_unique 
                    mu_target = xc_normed
                    c_target = c_unique

                    class_mu_update_op = tf.scatter_update(class_mu, labels_target, mu_target) 
                    class_log_sigma_sq_update_op = tf.scatter_update(class_log_sigma_sq, labels_target, -tf.log(c_target))
                    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, class_mu_update_op)
                    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, class_log_sigma_sq_update_op)
            else:
                labels_target = labels 
                weights_target = z_mu
                confidence_target = z_inv_sigma_sq

                weights_batch = tf.gather(class_mu, labels_target)
                diff_centers = weights_batch - weights_target
                diff_centers = alpha * diff_centers / appear_times # * confidence_target
                centers_update_op = tf.scatter_sub(class_mu, labels_target, diff_centers)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, centers_update_op)

            

    return loss

def mean_dev(z_mean, z_log_sigma_sq, labels, num_classes, global_step,  
        weight_decay, learning_rate, coef, alpha, multi_lr, reuse=None):
    num_features = z_mean.shape[1].value
    batch_size = tf.shape(z_mean)[0]
    weights_trainable = True if alpha is None else False
    with tf.variable_scope('ConditionalLoss', reuse=reuse):
        weights = tf.get_variable('weights', shape=(num_classes, num_features),
                regularizer=slim.l2_regularizer(0.0),
                initializer=slim.xavier_initializer(),
                # initializer=tf.truncated_normal_initializer(stddev=0.0),
                # initializer=tf.constant_initializer(0),
                trainable=False,
                dtype=tf.float32)
        # weights = batch_norm(tf.identity(weights))

        # z_inv_sigma_sq = tf.exp(-z_log_sigma_sq)

        labels_unique, unique_idx = tf.unique(labels)

        weights_batch = tf.gather(weights, labels_unique)

        xc = z_mean * z_inv_sigma_sq
        xc_unique = tf.unsorted_segment_sum(xc, unique_idx, tf.shape(labels_unique)[0])
        c_unique = tf.unsorted_segment_sum(z_inv_sigma_sq, unique_idx, tf.shape(labels_unique)[0])
        xc_normed = xc_unique / c_unique

        deviation = tf.reduce_sum(tf.square(xc_normed - weights_batch), axis=1)
        loss = coef * tf.reduce_mean(deviation) 

        # Update centers
        if weights not in tf.trainable_variables():
            if multi_lr:
                alpha = alpha * learning_rate

            diff_centers = weights_batch - xc_normed
            diff_centers = alpha * diff_centers
            centers_update_op = tf.scatter_sub(weights, labels_unique, diff_centers)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, centers_update_op)

    tfwatcher.insert('mean_sigma', tf.reduce_mean(tf.exp(z_log_sigma_sq)))

    return loss



def dim_pool(prelogits, confidence, label, num_classes, global_step, weight_decay, learning_rate,
                group_size, scale=16.0, m=1.0, alpha='auto', reuse=None):
    ''' Variant of AM-Softmax where weights are dynamically imprinted. '''
    num_features = prelogits.shape[1].value
    batch_size = tf.shape(prelogits)[0]
    with tf.variable_scope('DPLoss', reuse=reuse):
        weights = tf.get_variable('weights', shape=(num_classes, num_features),
                initializer=slim.xavier_initializer(),
                # initializer=tf.truncated_normal_initializer(stddev=0.0),
                # initializer=tf.constant_initializer(0),
                trainable=False,
                dtype=tf.float32)
        _scale = tf.get_variable('_scale', shape=(),
                regularizer=slim.l2_regularizer(1e-2),
                initializer=tf.constant_initializer(0.0),
                trainable=True,
                dtype=tf.float32)

        tf.add_to_collection('classifier_weights', weights)


        # Normalizing the vecotors
        prelogits_normed = tf.nn.l2_normalize(prelogits, dim=1)
        # prelogits_normed= prelogits
        weights_normed = tf.nn.l2_normalize(weights, dim=1)

        # Pooling
        # label_unique, unique_idx, unique_count = tf.unique_with_counts(label)
        # prelogits_mean = tf.unsorted_segment_sum(prelogits_normed, unique_idx, tf.shape(label_unique)[0])
        # prelogits_mean = prelogits_mean / tf.cast(tf.reshape(unique_count, [-1,1]), tf.float32)

        prelogits_mean = tf.reshape(prelogits_normed, [-1, group_size, num_features])

        confidence = tf.reshape(confidence, [-1, group_size, num_features])

        confidence = tf.nn.softmax(confidence, axis=1)
        # confidence = tf.nn.sigmoid(confidence)
        # confidence = confidenf.log(ce / tf.reduce_sum(confidence, axis=1, keep_dims=True)

        prelogits_mean = prelogits_mean * confidence
        prelogits_mean = tf.reduce_sum(prelogits_mean, axis=1)

        # Normalize Again
        # prelogits_mean = tf.nn.l2_normalize(prelogits_mean, dim=1)
        label_mean = tf.reshape(label, [-1, group_size])[:,0]
        
        print(prelogits_mean.shape)
        
        # centers = tf.gather(weights, label_mean)
        # loss = tf.reduce_mean(tf.reduce_sum(tf.square(prelogits_mean - centers), axis=1))

        # Update centers
        if False:
            if not weights in tf.trainable_variables():
                weights_target = prelogits_normedactivation
                label_target = label
                weights_batch = tf.gather(weights, label_target)
                diff_centers = weights_batch - weights_target
                unique_label, unique_idx, unique_count = tf.unique_with_counts(label_target)
                appear_times = tf.gather(unique_count, unique_idx)
                appear_times = tf.reshape(appear_times, [-1, 1])
                diff_centers = diff_centers / tf.cast(appear_times, tf.float32)
                diff_centers = alpha * diff_centers
                centers_update_op = tf.scatter_sub(weights, label_target, diff_centers)
                with tf.control_dependencies([centers_update_op]):
                    centers_update_op = tf.assign(weights, tf.nn.l2_normalize(weights,dim=1))
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, centers_update_op)


        tfwatcher.insert('meanp', tf.reduce_mean(prelogits_mean))

        return triplet_loss(label_mean, prelogits_mean, m)
        # return contrastive_loss(label_mean, prelogits_mean, m, normalize=False)

        # return loss


        # return am_softmax_imprint(prelogits_mean, label_mean, num_classes, global_step, 
        #         weight_decay, learning_rate, scale=scale, m=m, alpha=alpha, reuse=reuse)

        num_classes_batch = tf.shape(prelogits_mean)[0]

        dist_mat = euclidean_distance(prelogits_mean, tf.transpose(prelogits_mean), False)


        diag_mask = tf.eye(num_classes_batch, dtype=tf.bool)
        non_diag_mask = tf.logical_not(diag_mask)    

        label_mean = tf.reshape(label, [-1, group_size])[:,0]
        label_mat = tf.equal(label_mean[:,None], label_mean[None,:])
        label_mask_pos = tf.logical_and(non_diag_mask, label_mat)
        label_mask_neg = tf.logical_and(non_diag_mask, tf.logical_not(label_mat)) 

        
        dist_pos = tf.boolean_mask(dist_mat, label_mask_pos) 
        dist_neg = tf.boolean_mask(dist_mat, label_mask_neg)
        
        dist_pos = tf.reshape(dist_pos, [num_classes_batch, -1])
        dist_neg = tf.reshape(dist_neg, [num_classes_batch, -1])

        # return tf.reduce_mean(dist_pos)        


        # Hard Negative Mining
        dist_neg = -tf.reduce_max(-dist_neg, axis=1, keep_dims=True)
        
        loss = tf.nn.relu(dist_pos - dist_neg + m)
        loss = tf.reduce_mean(loss, name='DPLoss')


        return loss
