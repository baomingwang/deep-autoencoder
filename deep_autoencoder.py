# -*- coding: utf-8 -*-
"""
Created on Thu May 12 09:56:44 2016

@author: wangbm
"""

import tensorflow as tf
import numpy as np
from rbm import *

class Autoencoder(object):
    
    def __init__(self, input_num, layers, bp_learning_rate=0.01,  
                 bp_num_epoch=1, batch_size=128, data_type='float32'):
        self.input_num = input_num
        self.layers = layers
        self.n_layers = len(self.layers)
        self.bp_learning_rate = bp_learning_rate
        self.bp_num_epoch = bp_num_epoch
        self.batch_size = batch_size
        self.data_type = data_type
        self.W_list = []
        self.b_list = []
        self.a_list = []
        self.W_trained = []  
        self.b_trained = []
        self.a_trained = []
        self.W_eval = []
        self.b_eval = []
        self.a_eval = []
        self._initialize_weight()

    def _initialize_weight(self):
        '''
        randomly initialize weights for autoencoder
        '''
        self.W_list.append(np.random.random([self.input_num, self.layers[0]]).astype(self.data_type)/100.)
        self.b_list.append(np.zeros([self.layers[0]]).astype(self.data_type))
        for i in range(self.n_layers-1):
            self.W_list.append(np.random.random([self.layers[i], self.layers[i+1]]).astype(self.data_type)/100.)
            self.b_list.append(np.zeros([self.layers[i+1]]).astype(self.data_type))
        self.a_list.append(np.zeros([self.input_num]).astype(self.data_type))
        for j in range(self.n_layers-1):
            self.a_list.append(np.zeros([self.layers[j]]).astype(self.data_type))

    def backprop(self, train_set):
        '''
        use back propagation to train the model
        '''
        print 'Now begin to train the model with back propagation:'
        m, _ = train_set.shape
        self.num_per_epoch = m / self.batch_size         
        train_batch = tf.placeholder(self.data_type, [None, self.input_num])        
        logits = self._build_model(train_batch)
        loss = self._loss(logits, train_batch)
        train_op = self._training(loss)
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            for k in range(self.bp_num_epoch):
                for i in range(self.num_per_epoch):
                    _, cost = sess.run([train_op, loss], feed_dict = self._feed_build(train_batch, train_set, i))
                print 'loss for epoch %d: %e' %(k, cost)
            for j in range(self.n_layers):
                self.W_eval.append(self.W_trained[j].eval())
                self.b_eval.append(self.b_trained[j].eval())
                self.a_eval.append(self.a_trained[j].eval())
            for j in range(self.n_layers):
                self.W_eval.append(self.W_trained[j+self.n_layers].eval())
                    
    def _feed_build(self, train_batch, train_set, i):
        batch = next_batch(train_set, i, self.batch_size)
        feed_dict = {train_batch: batch}
        return feed_dict    
    
    def _build_model(self, train_batch):
        '''
        assemble the first and second half of the model
        '''
        middle_layer = self._make_encoder(train_batch)
        last_layer = self._make_decoder(middle_layer)
        return last_layer
        
    def _make_encoder(self, train_batch):
        '''
        build the first half of the model
        '''        
        encoder = []
        encoder.append(train_batch)    
        for i in range(self.n_layers):
            with tf.name_scope('encoder'+str(i)):
                self.W_trained.append(tf.Variable(self.W_list[i], name = 'weights'))
                self.b_trained.append(tf.Variable(self.b_list[i], name = 'biases'))
                encoder.append(tf.sigmoid(self.b_trained[i] + tf.matmul(encoder[i], self.W_trained[i])))
        return encoder[self.n_layers]
                
    
    def _make_decoder(self, middle_layer):
        '''
        build the second half of the model
        '''         
        decoder = []
        decoder.append(middle_layer)        
        for i in range(self.n_layers):
            with tf.name_scope('decoder'+str(i)):
                self.W_trained.append(tf.Variable(self.W_list[self.n_layers-i-1], name = 'weights'))
                self.a_trained.append(tf.Variable(self.a_list[self.n_layers-i-1], name = 'biases'))
                decoder.append(tf.sigmoid(self.a_trained[i] + tf.matmul(decoder[i], self.W_trained[i+self.n_layers], transpose_b = True)))
        return decoder[self.n_layers]        
    
    def _loss(self, logits, labels):
        loss = tf.nn.l2_loss(logits-labels)
        return loss
    
    def _training(self, loss):
        '''
        assign optimizer and objective function
        '''         
        optimizer = tf.train.GradientDescentOptimizer(self.bp_learning_rate)
        train_op = optimizer.minimize(loss)
        return train_op
        
    def get_para(self):
        return self.W_eval, self.b_eval, self.a_eval

class DeepAutoencoder(Autoencoder):
    
    def __init__(self,input_num, layers, rbm_learning_rate=0, rbm_num_epoch=1, momentum=0,
		bp_learning_rate=0.01,bp_num_epoch=1, batch_size=128, data_type='float32'):
        if momentum == 0:
            self.momentum = []            
            for _ in range(len(layers)):            
                self.momentum.append(0.5)
        else:
            self.momentum = momentum
        if rbm_learning_rate == 0:
            self.rbm_learning_rate = []            
            for _ in range(len(layers)):            
                self.rbm_learning_rate.append(0.1)
        else:
            self.rbm_learning_rate = rbm_learning_rate
        self.rbm_num_epoch = rbm_num_epoch
        self.rbm_list = []
        super(DeepAutoencoder, self).__init__(input_num, layers, bp_learning_rate, bp_num_epoch, batch_size, data_type)  
        	
    def _initialize_weight(self):        
        '''
        initialize weights trained by separate rbms
        ''' 
        self.rbm_list.append(RBM(self.input_num, self.layers[0], self.rbm_num_epoch, 
                                 self.momentum[0], self.rbm_learning_rate[0], self.batch_size, self.data_type))            
        for i in range(self.n_layers-1):
            self.rbm_list.append(RBM(self.layers[i], self.layers[i+1], self.rbm_num_epoch,
                            self.momentum[i], self.rbm_learning_rate[i], self.batch_size, self.data_type))

    def pretrain(self, train_set):
        '''
        implement the pretaining process
        this function must be called before backprop() when using DeepAutoencoder
        '''         
        print 'Now begin to pretrain the model with separate rbm:'
        if not cmp(train_set.dtype, self.data_type):
            train_set.dtype = self.data_type        
        next_train = train_set        
        for i, rboltz in enumerate(self.rbm_list):
            next_train = self._pretrain_and_get_para(rboltz, next_train)

    def _pretrain_and_get_para(self, rboltz, next_train):
        '''
        save the weights during pretraining
        return the hidden layer to be the input of next rbm
        '''         
        output, W_out, a_out, b_out = rboltz.fit(next_train)
        self.W_list.append(W_out)
        self.a_list.append(a_out)
        self.b_list.append(b_out)
        return output
        
if __name__ == '__main__':
    train_data = get_data('./train_mnist.mat', shuffle = True)

    ae = DeepAutoencoder(input_num=784, layers=[1000, 500, 250, 30], rbm_num_epoch=10, bp_num_epoch=10)

    ae.pretrain(train_data)
    ae.backprop(train_data)
