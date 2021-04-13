import tensorflow as tf

def merge_pairs(x,y):
    shape = [dim.value for dim in x.shape]
    shape_new = [-1] + shape[1:]
    z = tf.stack([x,y], axis=1)
    return tf.reshape(z, shape_new)
    

def split_pairs(x):
    shape = [dim.value for dim in x.shape]
    shape_new = [-1,2] + shape[1:]
    x_new = tf.reshape(x, shape_new)
    return x_new[:,0], x_new[:,1]
    

def random_interpolate(x, y):
    n = tf.shape(x)[0]
    nd = x.shape.ndims - 1
    r_shape = [n] + [1]*nd
    ratios = tf.random_uniform(r_shape)
    return ratios * x + (1-ratios) * y
