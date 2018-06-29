import numpy as np
import tensorflow as tf

vgg=np.load('vgg16.npy', encoding='latin1')

vgg_dict=vgg.item()

init = tf.constant_initializer(value=vgg_dict['conv5_1'][0],dtype=tf.float32)
shape = vgg_dict['conv5_1'][0].shape

var = tf.get_variable(name="filter", initializer=init, shape=shape)

shape = vgg_dict['conv5_1'][0].shape

if not tf.get_variable_scope().reuse:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), 5e-4 ,name='weight_loss')
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_decay)


print(a)