from __future__ import division

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import logging
import pickle
import os

logger = logging.getLogger('networks')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class VGG16:
    def __init__(self, learning_rate=1e-04):
        self.input_tensor = None
        self.label_tensor = None
        self.layer_feat = None
        self.loss = None
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    def read_original_weights(self, tf_session, path='./models/train_weights.txt', is_train=False):
        """
        original txt file contains
        """

        init = tf.global_variables_initializer()
        tf_session.run(init)
        logger.info('all global variables initialized')
	
        if not os.path.exists(path):
	        print("////////////////// * NO PRE_TRAINED WEIGHTS * //////////////////")
	        return 
	
        weights = pickle.load(open(path, 'rb'))
        keys = weights.keys()

        for var in tf.trainable_variables():
            if var.name in keys:
                val = weights[var.name]
                val = np.reshape(val, var.shape.as_list())
                tf_session.run(var.assign(val))
                logger.info('%s : original weights assigned. [0]=%s' % (var.name, str(val[0])[:20]))
            else:
                continue

        print(tf_session.run(tf.report_uninitialized_variables()))
    def save_train_model(self, tf_session, save_dir):
        weights = {}
        for var in tf.trainable_variables():
            weights[var.name] = tf_session.run(var)
        f = open(save_dir, 'wb')
        pickle.dump(weights, f)

    def create_network(self, input_tensor, label_tensor, is_training, reuse=None):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor

        # feature extractor - convolutions
        net = slim.convolution(input_tensor, 64, [3, 3], 1, padding='SAME', scope='conv1',
                               activation_fn=tf.nn.relu)

        net = slim.batch_norm(net)
        
        net = slim.convolution(net, 64, [3, 3], 1, padding='SAME', scope='conv2',
                               activation_fn=tf.nn.relu) 
        net = slim.batch_norm(net)

        net = slim.pool(net, [2, 2], 'MAX', stride=2, padding='SAME', scope='pool1')
        
        net = slim.convolution(net, 128, [3, 3], 1, padding='SAME', scope='conv3',
                               activation_fn=tf.nn.relu)
        
        net = slim.batch_norm(net)
        net = slim.convolution(net, 128, [3, 3], 1, padding='SAME', scope='conv4',
                               activation_fn=tf.nn.relu)
        net = slim.batch_norm(net)
       
        net = slim.pool(net, [2, 2], 'MAX', stride=2, padding='SAME', scope='pool2')


        net = slim.convolution(net, 256, [3, 3], 1, padding='SAME', scope='conv5',
                               activation_fn=tf.nn.relu)

        net = slim.batch_norm(net)
        net = slim.convolution(net, 256, [3, 3], 1, padding='SAME', scope='conv6',
                               activation_fn=tf.nn.relu)
        net = slim.batch_norm(net)
        net = slim.convolution(net, 256, [3, 3], 1, padding='SAME', scope='conv7',
                               activation_fn=tf.nn.relu)
        net = slim.batch_norm(net)
        net = slim.pool(net, [2, 2], 'MAX', stride=2, padding='SAME', scope='pool4')
        

        
        net = slim.convolution(net, 512, [3, 3], 1, padding='SAME', scope='conv8',
                               activation_fn=tf.nn.relu)
        net = slim.batch_norm(net)
        net = slim.convolution(net, 512, [3, 3], 1, padding='SAME', scope='conv9',
                               activation_fn=tf.nn.relu)
        net = slim.batch_norm(net)
        net = slim.convolution(net, 512, [3, 3], 1, padding='SAME', scope='conv10',
                               activation_fn=tf.nn.relu)
        net = slim.batch_norm(net)
        net = slim.pool(net, [2, 2], 'MAX', stride=2, padding='SAME', scope='pool5')


        net = slim.convolution(net, 512, [3, 3], 1, padding='SAME', scope='conv11',
                               activation_fn=tf.nn.relu)
        net = slim.batch_norm(net)
        net = slim.convolution(net, 512, [3, 3], 1, padding='SAME', scope='conv12',
                               activation_fn=tf.nn.relu)
        net = slim.batch_norm(net)
        net = slim.convolution(net, 512, [3, 3], 1, padding='SAME', scope='conv13',
                               activation_fn=tf.nn.relu)

        net = slim.batch_norm(net)
        
        net = slim.convolution(net, 1024, [3,3], 1, padding='SAME', scope='conv14',
                                activation_fn=tf.nn.relu)
        net = slim.batch_norm(net)
        net = slim.convolution(net, 1024, [3,3], 1, padding='SAME', scope='conv15',
                                activation_fn=tf.nn.relu)
        net = slim.batch_norm(net)
        net = slim.convolution(net, 1024, [3,3], 1, padding='SAME', scope='conv16',
                                activation_fn=tf.nn.relu)
        net = slim.batch_norm(net)

        net = slim.pool(net, [2,2], 'MAX', stride=2, padding='SAME', scope='pool6')
        
        self.layer_feat = net


        net = slim.convolution(net, 512, [4,4], 1, padding='VALID', scope='fc1',
                                activation_fn=tf.nn.relu)
        net = slim.convolution(net, 100, [1,1], 1, padding='VALID', scope='fc2',
                                activation_fn=tf.nn.relu)
        output = slim.convolution(net, 5, [1,1], 1, padding='VALID', scope='fc3',
                                activation_fn=None)

        output = flatten_convolution(output)
        
        self.output = tf.nn.softmax(output)

        # losses
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_tensor, logits=output)
        print(self.loss)
        
        wt_var = [var for var in tf.trainable_variables() if 'weight' in var.name]
        self.wt_decay = tf.reduce_sum([tf.nn.l2_loss(var) for var in wt_var])/64.
        self.wt_decay *=0.001
        
        self.loss += self.wt_decay
        self.opt = self.optimizer.minimize(self.loss)

def flatten_convolution(tensor_in):
    tendor_in_shape = tensor_in.get_shape()
    tensor_in_flat = tf.reshape(tensor_in, [tendor_in_shape[0].value or -1, np.prod(tendor_in_shape[1:]).value])
    return tensor_in_flat


if __name__ == '__main__':
    input_node = tf.placeholder(tf.float32, shape=(None, 128, 128, 3), name='patch')
    tensor_lb = tf.placeholder(tf.int32, shape=(None, 4), name='label')    # 
    is_training = tf.placeholder(tf.bool, name='is_training')

    model = VGG16()
    model.create_network(input_node, tensor_lb, is_training)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        # load all pretrained weights
        model.read_original_weights(sess)

        # zero input
        zeros = np.zeros(shape=(1, 112, 112, 3), dtype=np.float32)
        zeros_out = sess.run(model.layer_feat, feed_dict={input_node: zeros})
        pass
