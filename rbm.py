# -*- coding: utf-8 -*-
"""
Created on Fri May 06 10:25:02 2016

@author: wangbm
"""

import tensorflow as tf
import numpy as np
import h5py

#flags = tf.app.flags
#FLAGS = flags.FLAGS

#flags.DEFINE_string('summaries_dir', '/tmp/rbm', 'Summaries dir')
def get_data(data_dir, batch_size=128, shuffle=True):
    f = h5py.File(data_dir)
    variable_name = f.keys()[0]
    variable = f[variable_name][:]
    variable = np.transpose(variable)
    if shuffle == True:
        np.random.shuffle(variable)
    return variable
    
def next_batch(variable, batch_num, batch_size = 128):   
    n, m = variable.shape        
    num_per_batch = n / batch_size
    if batch_num < num_per_batch:
        data = variable[(batch_num*batch_size):((batch_num+1)*batch_size), :]
        return data
    else:
        pass
    
class RBM(object):
    def __init__(self, visiable_num, hidden_num, num_epoch=10, momentum=0.5, learning_rate=0.1, batch_size=100, data_type='float32'):
        self.visiable_num = visiable_num
        self.hidden_num = hidden_num
        self.num_epoch = num_epoch
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.P_h1 = None
        self.P_v2 = None
        self.P_h2 = None
        self.B_h1 = None
        self.W = None
        self.a = None
        self.b = None
        self.W_update = None
        self.a_update = None
        self.b_update = None
        self.data_type = data_type
        self.tf_session = None
        self.weightcost = 0.0002
        self.W_update = tf.zeros([self.visiable_num, self.hidden_num])
        self.a_update = tf.zeros([self.visiable_num])
        self.b_update = tf.zeros([self.hidden_num])
        self.cost = tf.Variable(0.)
        
    def _build_model(self):
        '''
        define the model graph
        '''
        self.train_batch = tf.placeholder(self.data_type, [self.batch_size, self.visiable_num], name='input')
        self.W = tf.Variable(np.random.rand(self.visiable_num, self.hidden_num)*0.1, dtype=self.data_type, name='weight')
        self.a = tf.Variable(np.zeros([self.visiable_num]), dtype=self.data_type, name='visable_biases')
        self.b = tf.Variable(np.zeros([self.hidden_num]), dtype=self.data_type, name='hidden_biases')
        
        self.P_h1 = tf.sigmoid(self.b + tf.matmul(self.train_batch, self.W), name='prob_hidden_1')
        self.B_h1 = tf.convert_to_tensor((tf.cast(tf.greater(self.P_h1, tf.cast(tf.Variable(np.random.rand(self.batch_size, self.hidden_num)), self.data_type)), self.data_type)), name='sampe_hidden_1')
        self.P_v2 = tf.sigmoid(self.a + tf.matmul(self.B_h1, self.W, transpose_b=True), name='prob_visable_2')
        self.P_h2 = tf.sigmoid(self.b + tf.matmul(self.P_v2, self.W), name='prob_hidden_2')
        with tf.name_scope('W_update'):
            self.W_update = self.momentum * self.W_update + self.learning_rate * \
                            ((tf.matmul(self.train_batch, self.P_h1, transpose_a=True) - tf.matmul(self.P_v2, self.P_h2, transpose_a=True)) / self.batch_size - self.weightcost*self.W)
        with tf.name_scope('a_update'):
            self.a_update = self.momentum * self.a_update + self.learning_rate * \
                            (tf.reduce_mean(self.train_batch, reduction_indices=[0]) - tf.reduce_mean(self.P_v2, reduction_indices=[0]))
        with tf.name_scope('b_update'):                    
            self.b_update = self.momentum * self.b_update + self.learning_rate * \
                            (tf.reduce_mean(self.P_h1, reduction_indices=[0]) - tf.reduce_mean(self.P_h2, reduction_indices=[0]))
        with tf.name_scope('cost'):
            #self.cost = tf.reduce_sum(tf.reduce_sum(tf.square(self.train_batch-self.P_v2)))
            self.update_cost = self.cost.assign_add(tf.reduce_sum(tf.reduce_sum(tf.square(self.train_batch-self.P_v2))))
        #self.summ = tf.scalar_summary('cost', self.cost)        
        self.update_W = self.W.assign_add(self.W_update)
        self.update_b = self.b.assign_add(self.b_update)
        self.update_a = self.a.assign_add(self.a_update)
                        
    def fit(self, train_set):     
        '''
        feed the input of graph and begin to train 
        ---------------------------------------------------------------------------
        INPUT:
            train_set: nparray, [data_num, dimensionality]
        
        OUTPUT:
            next_train: values of visible layer after training
            W_eval, a_eval, b_eval: weight and biases
        ---------------------------------------------------------------------------
        '''    
        print 'Now begin to train the rbm model:'
        self._build_model()
        m, _ = train_set.shape
        num_per_epoch = m / self.batch_size
        if not cmp(train_set.dtype, self.data_type):
            train_set.dtype = self.data_type        
        #merged = tf.merge_all_summaries()
        init = tf.initialize_all_variables()        
        with tf.Session() as self.tf_session:
            self.tf_session.run(init)
            #train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/try', self.tf_session.graph)
            for mn in range(self.num_epoch):
                reset = self.cost.assign(0.)
                self.tf_session.run(reset)
                if mn > 4:
                    self.momentum = 0.9
                if mn > 80:
                    self.learning_rate = 0.03
                for i in range(num_per_epoch):
                    #_,_,_,cost,summ = self.tf_session.run([self.update_W, self.update_b, self.update_a, self.cost, self.summ], feed_dict = self._feed_build(train_set, i))
                    self.tf_session.run([self.update_W, self.update_b, self.update_a, self.update_cost], feed_dict = self._feed_build(train_set, i))
                    #train_writer.add_summary(summ, i+mn)
                print 'loss for epoch %d: %e' %(mn, self.cost.eval())
            W_eval = self.W.eval()
            a_eval = self.a.eval()
            b_eval = self.b.eval()
        next_train = self.eval_model_output(train_set, W_eval, b_eval)
        return next_train, W_eval, a_eval, b_eval
    
    def _feed_build(self, train_set, i):
        '''
        used to create the feed dictionary
        '''
        batch = next_batch(train_set, i, self.batch_size)
        feed_dict = {self.train_batch: batch}
        return feed_dict

    def eval_model_output(self, train_set, W_eval, b_eval):
        '''
        use the existing weight to predict the visible layer 
        ---------------------------------------------------------------------------
        INPUT:
            train_set: nparray, [data_num, dimensionality]
            W_eval, b_eval: weight and biases
        
        OUTPUT:
            output: values of visible layers
        ---------------------------------------------------------------------------
        '''         
        f = lambda x: 1 / (1 + np.e**(-x)) 
        P = f(train_set.dot(W_eval) + b_eval)
        output = np.float32(P > np.random.random(P.shape))
        return output

if __name__ == '__main__':
    train_data = get_data('./train_mnist.mat', shuffle = True)
    
    rbm1 = RBM(visiable_num=784, hidden_num=1000)

    next_train, W_out, a_out, b_out = rbm1.fit(train_data)
