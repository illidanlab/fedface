import tensorflow as tf
import numpy as np

def rank_accuracy(logits, label, k=1):
    batch_size = tf.shape(logits)[0]
    _, arg_top = tf.nn.top_k(logits, k)
    label = tf.cast(label, tf.int32)
    label = tf.reshape(label, [batch_size, 1])
    label = tf.tile(label, [1, k])
    correct = tf.reduce_any(tf.equal(label, arg_top), axis=1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy

def cosine_pair(x1, x2):
    #assert x1.shape == x2.shape
    eps = 1e-10
    x1_norm = tf.sqrt(tf.reduce_sum(tf.square(x1), axis=1, keepdims=True))
    x2_norm = tf.sqrt(tf.reduce_sum(tf.square(x2), axis=1, keepdims=True))
    x1 = x1 / (x1_norm + eps)
    x2 = x2 / (x2_norm + eps)
    dist = tf.reduce_sum(x1 * x2, axis=1)
    return dist

def convert_to_classifier(x1, x2):
    distance = tf.reduce_sum(tf.square(x1 - x2), axis=1)
    threshold = 0.99
    score = tf.where(distance > threshold,
                    0.5 + ((distance - threshold) * 0.5) / (4.0 - threshold),
                    0.5 * distance / threshold)
    reverse_score = 1.0 - score
    return tf.transpose(tf.stack([reverse_score, score]))

def cosine_pair_np(x1, x2):
    assert x1.shape == x2.shape
    epsilon = 1e-10
    x1_norm = np.sqrt(np.sum(np.square(x1), axis=1, keepdims=True))
    x2_norm = np.sqrt(np.sum(np.square(x2), axis=1, keepdims=True))
    x1 = x1 / (x1_norm+epsilon)
    x2 = x2 / (x2_norm+epsilon)
    dist = np.sum(x1 * x2, axis=1)
    return dist

