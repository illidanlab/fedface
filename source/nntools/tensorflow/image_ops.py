import tensorflow as tf

def std_normalize(images, stop_gradient=True):
    mean, var = tf.nn.moments(images, axes=[1,2], keep_dims=True)
    std = tf.sqrt(var + 1e-8)
    images = (images - tf.stop_gradient(mean)) / tf.stop_gradient(std)
    return images


def image_grid(images, size):
    m, n = size
    h, w, c = images.shape[1:4]
    images = tf.reshape(images, [m, n, h, w, c])
    images = tf.transpose(images, [0, 2, 1, 3, 4])
    image_grid = tf.reshape(images, [1, m*h, n*w, c])
    return image_grid
