
# # GradCAM Visualization Demo with VGG16
# 
# Requirement:
# 
# * GPU Memory: 6GB or higher

# Replace vanila relu to guided relu to get guided backpropagation.
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops.relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))


def lb2onehot(labels, num_class=5):
    result = tf.zeros([len(labels), num_class])
    for idx, lb in enumerate(labels):
        result[idx, lb] = 1
    return result


import numpy as np
import utils
import pickle

# Create mini-batch for demo
# Get normalized input. VGG network handles the normalized image internally. 
test_image = pickle.load(open('test_img.txt', 'rb'))
test, _test_lb = [], []
for lb in range(5):
    test.extend(test_image[lb])
    num_lb = len(test_image[lb])
    _test_lb.extend([lb]*num_lb)

test = np.array(test)
test_lb = np.array(_test_lb)


# Create tensorflow graph for evaluation
from networks import VGG16
batch_size = 50
eval_graph = tf.Graph()
with eval_graph.as_default():
    with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
    
        images = tf.placeholder(tf.float32, [batch_size, 128, 128, 3])
        labels = tf.placeholder(tf.int32, [batch_size, ])        
        print(images, labels)
        is_training = tf.placeholder(tf.bool, name='is_training')
        lr = tf.placeholder(tf.float32, [], name='learning_rate')

        vgg = VGG16()
        vgg.create_network(images, labels, is_training)
        
        cost = vgg.loss
        print('cost:', cost)
        
        # gradient for partial linearization. We only care about target visualization class. 
        y_c = tf.reduce_sum(tf.multiply(vgg.output_logit, tf.one_hot(labels,5)), axis=1)
        print('y_c:', y_c)
        # Get last convolutional layer gradient for generating gradCAM visualization
        target_conv_layer = vgg.layer_feat
        target_conv_layer2 = vgg.layer_feat_
        target_conv_layer3 = vgg.layer_feat__

        target_conv_layer_grad = tf.gradients(y_c, target_conv_layer)[0]
        target_conv_layer_grad2 = tf.gradients(y_c, target_conv_layer2)[0]
        target_conv_layer_grad3 = tf.gradients(y_c, target_conv_layer3)[0]
        # Guided backpropagtion back to input layer
        gb_grad = tf.gradients(cost, images)[0]
        
# Run tensorflow 

with tf.Session(graph=eval_graph) as sess:    
    vgg.read_original_weights(sess)
    mis_class = []
    lb_pair = []
    pb_list = []
    for iter in range(50):
        prob = sess.run(vgg.output, feed_dict={images: test[iter*batch_size:(iter+1)*batch_size]})
        
        
        gb_grad_value, target_conv_layer_value, target_conv_layer_value2, target_conv_layer_value3, target_conv_layer_grad_value, target_conv_layer_grad_value2, target_conv_layer_grad_value3 = sess.run([gb_grad, target_conv_layer, target_conv_layer2, target_conv_layer3, target_conv_layer_grad, target_conv_layer_grad2, target_conv_layer_grad3], 
            feed_dict={images: test[iter*batch_size:(iter+1)*batch_size], 
                labels: test_lb[iter*batch_size:(iter+1)*batch_size]})
        

        #gb_grad_value, target_conv_layer_value, target_conv_layer_grad_value = sess.run([gb_grad, target_conv_layer, target_conv_layer_grad], feed_dict = {images: test[iter*batch_size:(iter+1)*batch_size], labels: test_lb[iter*batch_size: (iter+1)*batch_size]})

        for i in range(batch_size):
            if test_lb[iter*batch_size + i] == np.argmax(prob[i]):
                mis_class.append(0)
                is_mis = False
            else:
                mis_class.append(1)
                is_mis = True
            
            lb_pair.append((test_lb[iter*batch_size+i], np.argmax(prob[i])))
            pb_list.append(prob[i])

            if is_mis:
                print(iter*batch_size+i, 'th image was misclassified\n Label:\t', 
                    test_lb[iter*batch_size+i], 'Pred:\t', np.argmax(prob[i]), 'prob_list:\t', prob[i])
            
            utils.save_result(test[iter*batch_size+i], target_conv_layer_value[i], target_conv_layer_grad_value[i], gb_grad_value[i], iter*batch_size+i) 
            utils.save_result(test[iter*batch_size+i], target_conv_layer_value2[i], target_conv_layer_grad_value2[i], gb_grad_value[i],iter*batch_size+i, 1) 
            utils.save_result(test[iter*batch_size+i], target_conv_layer_value3[i], target_conv_layer_grad_value3[i], gb_grad_value[i], iter*batch_size+i,2)

    pickle.dump(mis_class, open('mis_class.txt','wb'))
    pickle.dump(lb_pair, open('lb_pair.txt','wb'))
    pickle.dump(pb_list, open('pb_list.txt', 'wb'))
