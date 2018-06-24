from __future__ import division

import os
import random
import sys
import logging

import cv2
import fire
import numpy as np
import tensorflow as tf
import tensorflow.contrib.metrics as metrics
import time
import pickle

import commons
from configs import VGGConf
from networks import VGG16
from pystopwatch import StopWatchManager

_log_level = logging.DEBUG
_logger = logging.getLogger('ADNetRunner')
_logger.setLevel(_log_level)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(_log_level)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
_logger.addHandler(ch)


class VGGRunner:
    MAX_BATCHSIZE = 512

    def __init__(self):
        self.tensor_input = tf.placeholder(tf.float32, shape=(None, 128, 128, 3), name='input')
        self.tensor_lb = tf.placeholder(tf.int32, shape=(None, ), name='lb_action')
        self.tensor_is_training = tf.placeholder(tf.bool, name='is_training')
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')
        config = tf.ConfigProto() #log_device_placement=True)

        self.persistent_sess = tf.Session(config=config)

        self.model = VGG16(self.learning_rate_placeholder)
        self.model.create_network(self.tensor_input, self.tensor_lb, self.tensor_is_training)
        #self.persistent_sess.run(tf.global_variables_initializer())
        self.model.read_original_weights(self.persistent_sess)
        self.iteration = 0
        self.imgwh = None

        self.loss = []
        self.train, self.test =[], []
        self.train_lb, self.test_lb = [], []

        self.stopwatch = StopWatchManager()

    def read_train_test(self, data_path, is_train=True):
        path = commons.get_image_pathes(data_path)
        images, labels = commons.read_image_lb_path(path, is_train)
        train_f_num = VGGConf.g()['train_folder_num']
        test_f_num = VGGConf.g()['test_folder_num']
        f_num = train_f_num + test_f_num
        print(f_num)
        assert f_num == len(images)
        assert len(images) == len(labels)

        for idx in range(f_num):
            if idx < train_f_num:
                if not is_train: continue
                self.train.extend(images[idx])
                self.train_lb.extend(labels[idx])
                print("Train data is added")
            else:
                self.test.extend(images[idx])
                self.test_lb.extend(labels[idx])
                print("Test data is added")
        
        _temp = [0,0,0,0,0]
        for i, a in enumerate(self.train_lb):
            _temp[int(a)]+=1
        print("Trainind Data Label Distribution: ", _temp)
        print("READING DATA IS COMPLETED")

    def label_index(self):
        label_array = np.array(self.test_lb)
        self.lb_idx = []
        for i in range(5):
            self.lb_idx.append(list(np.where(label_array==i)))
    

    def run_train(self, data_path='./data'):
        train = pickle.load(open('train_img.txt','rb'))
        self.train, self.train_lb = [], []
        for lb in range(5):
            self.train.extend(train[lb])
            num_lb = len(train[lb])
            self.train_lb.extend([lb]*num_lb)
        #self.read_train_test(data_path)
        train_num = len(self.train)
        train_index = list(range(train_num))

        EPOCH = VGGConf.g()['epoch']
        BATCH_SIZE = VGGConf.g()['minibatch_size']
        BATCH_NUM = int(len(train_index)/BATCH_SIZE) - 1
        learning_rate = VGGConf.g()['learning_rate']
        print("START TO TRAIN")
        for epoch_iter in range(EPOCH):
            iter = 0
            #if epoch_iter != 0 and epoch_iter % 100 ==0 : learning_rate *= 0.5
            random.shuffle(train_index)
            #for i in range(5): random.shuffle(self.lb_idx[i])
            for batch_iter in range(BATCH_NUM):
                batch_img = commons.choices_by_idx(self.train, train_index[iter*BATCH_SIZE: (iter+1)*BATCH_SIZE])
                batch_lb = commons.choices_by_idx(self.train_lb, train_index[iter*BATCH_SIZE: (iter+1)*BATCH_SIZE])

                _, loss = self.persistent_sess.run(
                    [self.model.opt, self.model.loss],
                    feed_dict={
                        self.tensor_input: np.array(batch_img)/255.,
                        self.tensor_lb: batch_lb,
                        self.learning_rate_placeholder: learning_rate,
                        self.tensor_is_training: True
                    }
                )

                print("{}/{} BATCH LOSS: {}".format(batch_iter, BATCH_NUM, np.sum(loss)))
                iter +=1

        self.model.save_train_model(self.persistent_sess, VGGConf.g()['save_weight_dir'])
        self.run_test(weight_load=False)

    def run_test(self, weight_load=True, data_path='./data'):
        test = pickle.load(open('test_img.txt','rb'))
        self.test, self.test_lb = [], []
        for lb in range(5):
            self.test.extend(test[lb])
            num_lb = len(test[lb])
            self.test_lb.extend([lb]*num_lb)

        MAX_BATCHSIZE = self.MAX_BATCHSIZE
        if weight_load:
            #self.read_train_test(data_path, is_train=False)
            self.model.read_original_weights(self.persistent_sess)
        test_num = len(self.test)
        if test_num % self.MAX_BATCHSIZE == 0 : 
            test_batch_num = int(test_num / self.MAX_BATCHSIZE)
        else:
            test_batch_num = int(test_num / self.MAX_BATCHSIZE) + 1

        test_output = []

        test_idx = list(range(test_num))
        test_iter = 0 
        print("START TO TEST")
        for batch_iter in range(test_batch_num):
            if batch_iter == test_batch_num - 1:
                batch_img = commons.choices_by_idx(self.test, test_idx[test_iter*MAX_BATCHSIZE:])
                batch_lb = commons.choices_by_idx(self.test_lb, test_idx[test_iter*MAX_BATCHSIZE:])
            else:
                batch_img = commons.choices_by_idx(self.test, test_idx[test_iter*MAX_BATCHSIZE: (test_iter+1)*MAX_BATCHSIZE])
                batch_lb = commons.choices_by_idx(self.test_lb, test_idx[test_iter*MAX_BATCHSIZE: (test_iter+1)*MAX_BATCHSIZE])

            output = self.persistent_sess.run(
                self.model.output, feed_dict={
                        self.tensor_input: np.array(batch_img)/255.,
                        self.tensor_lb: batch_lb,
                        self.tensor_is_training: False
                }
            )
            test_output.extend(output)
            test_iter +=1
        
        pickle.dump(self.test_lb, open('./gt.txt', 'wb'))
        pickle.dump(test_output, open('./pred.txt', 'wb'))

        accuracy = 0.0
        for idx, output in enumerate(test_output):
            max_idx = np.argmax(output)
            if max_idx == int(self.test_lb[idx]): accuracy+=1.0

        accuracy = accuracy / (len(self.test_lb))
        print("Test Accuracy: ", accuracy)


    def _get_features(self, samples):
        feats = []
        for batch in commons.chunker(samples, ADNetRunner.MAX_BATCHSIZE):
            feats_batch = self.persistent_sess.run(self.adnet.layer_feat, feed_dict={
                self.adnet.input_tensor: batch
            })
            feats.extend(feats_batch)
        return feats

        # train_fc_finetune_hem
        self._finetune_fc(
            img, pos_boxes, neg_boxes, pos_lb_action,
            ADNetConf.get()['initial_finetune']['learning_rate'],
            ADNetConf.get()['initial_finetune']['iter']
        )

        self.histories.append((pos_boxes, neg_boxes, pos_lb_action, np.copy(img), self.iteration))
        _logger.info('ADNetRunner.initial_finetune t=%.3f' % t)
        self.stopwatch.stop('initial_finetune')


    def __del__(self):
        self.persistent_sess.close()

if __name__ == '__main__':
    VGGConf.get('./conf/conf.yaml')

    random.seed(1258)
    np.random.seed(1258)
    tf.set_random_seed(1258)

    fire.Fire(VGGRunner)
