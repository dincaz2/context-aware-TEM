'''
Tensorflow implementation of Neural Factorization Machines as described in:
Xiangnan He, Tat-Seng Chua. Neural Factorization Machines for Sparse Predictive Analytics. In Proc. of SIGIR 2017.

This is a deep version of factorization machine and is more expressive than FM.

@author: 
Xiangnan He (xiangnanhe@gmail.com)
Lizi Liao (liaolizi.llz@gmail.com)

@references:
'''
import os

from TEM.factory import ensureDir
from TEM.load_node import NodeGenerator
from TEM.metrics import eval_model_pro

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# import sys
# import math
import numpy as np
# import scipy as sp
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, log_loss
from time import time, sleep
import argparse
tf = tf.compat.v1
# import zipfile
# from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
# import pandas

# from load_node import NodeGenerator
# from metrics import *
# from factory import *

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Extract Node Features.")
    # ++++++++++++++++++++++++++++basic setting+++++++++++++++++++++++++++++++++++
    parser.add_argument('--out_dir', nargs='?', default='nar',
                        help='Output directory in (FM or NFM)')
    parser.add_argument('--city', nargs='?', default='London',
                        help='City Name in (London, New_York_City, Singapore)')
    parser.add_argument('--cate', nargs='?', default='Attractions',
                        help='Category in (Attractions, Restaurant)')
    parser.add_argument('--data_type', nargs='?', default='attr',
                        help='Data Type in (attr, id_attr, id+attr).')
    # +++++++++++++++++++++++++++++ learning rate for two parts ++++++++++++++++++++++++++++++++++
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    # +++++++++++++++++++++++++++++ main params for tree part ++++++++++++++++++++++++++++++++++
    parser.add_argument('--num_boost_round', type=int, default=500,
                        help='num of boost round.')
    parser.add_argument('--max_depth', type=int, default=6,
                        help='num of max depth')
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++ main params for basic fm part ++++++++++++++++++++++++++++++++++
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=5120,
                        help='Batch size.')
    parser.add_argument('--batch_neg_sample', type=int, default=4,
                        help='Batch size.')
    # +++++++++++++++++++++++++++++ main params for fm embedding part ++++++++++++++++++++++++++++++++++
    parser.add_argument('--hidden_factor', type=int, default=20,
                        help='Number of hidden factors.')
    parser.add_argument('--feature_pooling', nargs='?', default='mean',
                        help='Specify a loss type (mean pooling or max pooling).')
    parser.add_argument('--merge_pooling', nargs='?', default='mean',
                        help='Specify a loss type (mean pooling or max pooling).')
    # +++++++++++++++++++++++++++++ main params for fm attention part ++++++++++++++++++++++++++++++++++
    parser.add_argument('--attention_factor', type=int, default=20,
                        help='Number of attention factors.')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Index of coefficient of sum of exp(A)')
    # +++++++++++++++++++++++++++++ other params for fm part ++++++++++++++++++++++++++++++++++
    parser.add_argument('--loss_type', nargs='?', default='log_loss',
                        help='Specify a loss type (square_loss or log_loss).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--early_stop', type=int, default=1,
                        help='Whether to perform early stop (0 or 1)')
    parser.add_argument('--reg_embedding', type=float, default=0,
                        help='Regularizer for embeddings.')
    parser.add_argument('--reg_bias', type=float, default=0.,
                        help='Regularizer for bias term')
    # +++++++++++++++++++++++++++++ other params for fm part ++++++++++++++++++++++++++++++++++
    parser.add_argument('--algorithm', type=int, default=0,
                        help='Whether to perform attention over the concatenated representations')
    parser.add_argument('--ui_flag', type=int, default=0,
                        help='Whether to perform early stop (0 or 1)')
    parser.add_argument('--bias_flag', type=int, default=0,
                        help='Whether to perform early stop (0 or 1)')
    parser.add_argument('--max_flag', type=int, default=1,
                        help='Whether to perform early stop (0 or 1)')
    return parser.parse_args()

class NAR(BaseEstimator, TransformerMixin):
    def __init__(self, ui_M, raw_M, nodes_M, args, random_seed=2017):
        self.ui_M = ui_M
        self.raw_M = raw_M
        self.nodes_M = nodes_M

        self.hidden_factor = args.hidden_factor

        self.feature_pooling = args.feature_pooling
        self.merge_pooling = args.merge_pooling

        self.attention_factor = args.attention_factor
        self.beta = args.beta

        self.batch_size = args.batch_size
        self.batch_neg_sample = args.batch_neg_sample
        self.loss_type = args.loss_type

        self.epoch = args.epoch
        self.random_seed = random_seed
        self.learning_rate = args.lr
        self.verbose = args.verbose
        self.early_stop = args.early_stop

        self.reg_embedding = args.reg_embedding
        self.reg_bias = args.reg_bias

        self.algorithm = args.algorithm
        self.ui_flag = args.ui_flag
        self.bias_flag = args.bias_flag
        self.max_flag = args.max_flag

        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.train_ui = tf.placeholder(tf.int32, shape=[None, 2]) # (None, 2)
            self.train_raw = tf.placeholder(tf.int32, shape=[None, None]) # (None, None)
            self.train_nodes = tf.placeholder(tf.int32, shape=[None, None]) # (None, None)

            self.train_labels = tf.placeholder(tf.float32, shape=[None, 1])  # None * 1

            # Variables.
            self.weights = self._initialize_weights()

            # Model.
            # _________ original representation part _____________
            # get the representations of the user-item interactions.
            ui_rep = tf.nn.embedding_lookup(self.weights['ui_embeddings'], self.train_ui) # (None, 2, hidden_factor)
            interaction_rep = tf.reduce_prod(ui_rep, axis=1, keepdims=True) # (None, 1, hidden_factor)

            # get the representations of the associated rules.
            nodes_rep = tf.nn.embedding_lookup(self.weights['node_embeddings'], self.train_nodes) # (None, nodes_M, hidden_factor)

            # _________ attentive representation part _____________
            # get the representations of the user-item interactions.
            if self.algorithm == 0:
                # input = (None, nodes_M, hidden_factor);
                # output = (None, hidden_factor)
                attention_rep = self._adding_attention(nodes_rep * interaction_rep, nodes_rep)
            elif self.algorithm == 1:
                n = tf.shape(nodes_rep)[1]
                # input = (None, nodes_M, 2*hidden_factor)
                # output = (None, hidden_factor)
                attention_rep = self._adding_attention(tf.concat([nodes_rep, tf.tile(interaction_rep, tf.stack([1, n, 1]))], axis=2), nodes_rep) # (None, hidden_factor)
            else:
                attention_rep = tf.reduce_sum(nodes_rep, 1)

            # _________ prediction part _____________
            # get the representations of the user-item interactions.
            if self.ui_flag == 0:
                # output = (None, hidden_factor)
                self.out = attention_rep
            else:
                # output = (None, 2*hidden_factor)
                self.out = tf.concat([attention_rep, tf.reduce_sum(interaction_rep, axis=1)], axis=1)


            self.out = tf.matmul(self.out, self.weights['prediction']) # (None, 1)
            bias = self.weights['bias'] * tf.ones_like(self.train_labels)  # (None, 1)

            # _________ raw bias part _____________
            # get the representations of the user-item interactions.
            if self.bias_flag == 0:
                # no any bias.
                self.out = tf.add_n([self.out, bias])
            elif self.bias_flag == 1:
                # adding bias for node features.
                nodes_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['nodes_bias'], self.train_nodes), 1) # (None, 1)
                self.out = tf.add_n([self.out, nodes_bias, bias])
            elif self.bias_flag == 2:
                # adding bias for raw features.
                raw_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['raw_bias'], self.train_raw), 1) # (None, 1)
                self.out = tf.add_n([self.out, raw_bias, bias])
            else:
                # adding bias for node & raw features.
                nodes_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['nodes_bias'], self.train_nodes), 1) # (None, 1)
                raw_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['raw_bias'], self.train_raw), 1) # (None, 1)
                self.out = tf.add_n([self.out, raw_bias, nodes_bias, bias])


            # calculate the log loss.
            self.out = tf.sigmoid(self.out)
            self.loss = tf.losses.log_loss(self.train_labels, self.out, epsilon=1e-15, scope=None)

            # add regularization terms.
            if self.reg_embedding > 0:
                self.loss = self.loss + self.reg_embedding * (tf.reduce_sum(tf.square(self.weights['ui_embeddings']))
                            + tf.reduce_sum(tf.square(self.weights['node_embeddings'])))

            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            #init = tf.initialize_all_variables()
            self.sess = tf.Session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape() # shape is an array of tf.Dimension
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim
                total_parameters += variable_parameters
            if self.verbose > 0:
                print(f"#params: {total_parameters}")

    def _adding_attention(self, rep, feature_rep):
        # tf.shape(rep) = (None, nodes_M, hidden_factor)
        # tf.shape(rep) = (None, nodes_M, hidden_factor)
        with tf.name_scope("adding_attention"):
            b = tf.shape(rep)[0] # batch_size (a.k.a., None)
            n = tf.shape(rep)[1] # rule_size (a.k.a., nodes_M)
            r = (self.algorithm + 1) * self.hidden_factor # embedding_size (a.k.a., hidden_factor)

            rep_att = tf.matmul(tf.reshape(rep, [-1,r]), self.weights['W_att']) + self.weights['b_att'] # (None * nodes_M, hidden_factor) * (hidden_factor, attention_factor) => (None * nodes_M, attention_factor)
            rep_att = tf.nn.relu(rep_att) # (None * nodes_M, attention_factor)

            rep_att = tf.reshape(tf.matmul(rep_att, self.weights['h_att']), [b, n]) # (None*nodes_M, attention_factor) * (attention_factor, 1) => (None, nodes_M)

            att_exp = tf.exp(rep_att) # (None, nodes_M)
            exp_sum = tf.reduce_sum(att_exp, axis=1, keepdims=True) # (None, 1)
            exp_sum = tf.pow(exp_sum, tf.constant(self.beta, tf.float32, [1]))

            A = tf.expand_dims(tf.div(att_exp, exp_sum), 2) # (None, nodes_M, 1)

            if self.max_flag == 0:
                A = tf.reduce_sum(A * feature_rep, 1) # (None, hidden_factor)
            else:
                A = tf.reduce_max(A * feature_rep, 1) # (None, hidden_factor)
            return A

    def _initialize_weights(self):
        all_weights = dict()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        all_weights['ui_embeddings'] = tf.Variable(tf.truncated_normal([self.ui_M+1, self.hidden_factor], 0.0, 0.1), # users_M * hidden_factor
                                                    name='ui_embeddings', dtype=tf.float32)
        all_weights['node_embeddings'] = tf.Variable(tf.truncated_normal([self.nodes_M+1, self.hidden_factor], 0.0, 0.1), # nodes_M * hidden_factor
                                                    name='node_embeddings', dtype=tf.float32)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        all_weights['prediction'] = tf.Variable(tf.truncated_normal([(1+self.ui_flag)*self.hidden_factor, 1], 0.0, 0.1), # hidden_factor * 1
                                                        name='prediction', dtype=tf.float32)

        all_weights['bias'] = tf.Variable(tf.constant(0.0, dtype=tf.float32), name='bias', dtype=tf.float32)  # 1 * 1
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        all_weights['W_att'] = tf.Variable(tf.truncated_normal(shape=[(1+self.algorithm)*self.hidden_factor, self.attention_factor], mean=0.0,
                                           stddev=tf.sqrt(tf.div(2.0, (1+self.algorithm)*self.hidden_factor + self.attention_factor))),name='attention_weights', dtype=tf.float32, trainable=True)
        all_weights['b_att'] = tf.Variable(tf.truncated_normal(shape=[1, self.attention_factor], mean=0.0,
                                           stddev=tf.sqrt(tf.div(2.0, self.hidden_factor + self.attention_factor))),name='attention_bias', dtype=tf.float32, trainable=True)
        all_weights['h_att'] = tf.Variable(tf.ones([self.attention_factor, 1]), name='attention_h', dtype=tf.float32)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        all_weights['raw_bias'] = tf.Variable(tf.truncated_normal([self.raw_M, 1], 0.0, 0.1),
                                                    name='nodes_bias', dtype=tf.float32)
        all_weights['nodes_bias'] = tf.Variable(tf.truncated_normal([self.nodes_M, 1], 0.0, 0.),
                                                    name='nodes_bias', dtype=tf.float32)
        return all_weights


    def train(self, Data_load):
        # Training epochs
        self.best_valid_logloss, self.best_logloss_epoch, self.best_test_logloss = 100., 0, 100.
        self.best_valid_ndcg, self.best_ndcg_epoch, self.best_test_ndcg = 0, 0, 0
        self.best_valid_hit, self.best_hit_epoch, self.best_test_hit = 0, 0, 0

        self.valid_epochs = [] # record validation score of each epoch
        for epoch in range(self.epoch):
            t0 = time()

            train_batches, train_num_loop = Data_load._generate_train_batch(self.batch_size, self.batch_neg_sample)

            for i in range(train_num_loop):

                x_ui, x_raw, x_node, y_list = Data_load._fetch_train_batch_data(train_batches, flag='train', current_loop=i)
                y_list.shape += (1,)

                feed_dict = {self.train_ui: x_ui, self.train_raw: x_raw, self.train_nodes: x_node, self.train_labels: y_list}
                temp_loss, training_optimizer = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
                # print('train %d over %d' % (i, train_num_loop))
            t1 = time()
            # train_hit, train_ndcg, train_loss = self.evaluate(Data_load, train_flag='train')
            train_hit, train_ndcg, train_loss = 0., 0., 0.
            t2 = time()
            valid_hit, valid_ndcg, valid_loss = self.evaluate(Data_load, train_flag='valid')
            t3 = time()
            test_hit, test_ndcg, test_loss = self.evaluate(Data_load, train_flag='test')
            t4 = time()

            # self.valid_epochs.append(valid_loss)
            self.valid_epochs.append(valid_ndcg)

            if self.best_valid_logloss > valid_loss:
                self.best_valid_logloss = valid_loss
                self.best_logloss_epoch = epoch
                self.best_test_logloss = test_loss

            if self.best_valid_ndcg < valid_ndcg:
                self.best_valid_ndcg = valid_ndcg
                self.best_ndcg_epoch = epoch
                self.best_test_ndcg = test_ndcg

            if self.best_valid_hit < valid_hit:
                self.best_valid_hit = valid_hit
                self.best_hit_epoch = epoch
                self.best_test_hit = test_hit

            # showProgress
            if self.verbose > 0 and epoch%self.verbose == 0:
                print("Epoch %d [%.1f s]\ttrain=[%.6f], valid=[%.6f, %.6f, %.6f], test=[%.6f, %.6f, %.6f]; time=[%.1fs, %.1fs, %.1fs]"
                    %(epoch+1, t1-t0, train_loss, valid_loss, valid_hit, valid_ndcg, test_loss, test_hit, test_ndcg, t2-t1, t3-t2, t4-t3))
                # print("Epoch %d [%.1f s]\ttrain=[%.6f, %.6f, ], valid=[%.6f, %.6f, %.6f], test=[%.6f, %.6f, %.6f] [%.1f s, %.1f]"
                #       %(epoch+1, t1-t0, t2-t1, training_loss, valid_loss, valid_ndcg, valid_auc, test_loss, test_ndcg, test_auc, t3-t2, t4-t3))
            if self.early_stop > 0 and self.eva_termination(self.valid_epochs, order_flag='inc'):
                break
        final_result_str = '\tEnd. valid=[%.6f, %.6f, %.6f], test=[%.6f, %.6f, %.6f], @epoch=%d' % (self.best_valid_logloss, self.best_valid_hit, self.best_valid_ndcg, self.best_test_logloss, self.best_test_hit, self.best_test_ndcg, self.best_logloss_epoch)
        print(final_result_str)
        # print('\tEnd. valid=[%.6f, %.6f] @epoch=%d' % (self.best_valid_ndcg, self.best_valid_auc, self.best_ndcg_epoch))
        # print('\t      test=[%.6f, %.6f] @epoch=%d' % (self.best_test_ndcg, self.best_test_auc, self.best_auc_epoch))
        # final_result = [self.best_valid_logloss, self.best_test_logloss, self.best_logloss_epoch]
        # final_valid = [self.best_valid_ndcg, self.best_valid_auc, self.best_ndcg_epoch]
        # final_test = [self.best_test_ndcg, self.best_test_auc, self.best_auc_epoch]
        # return final_valid, final_test
        return final_result_str

    def eva_termination(self, valid, order_flag='des'):
        if len(valid) > 5:
            if order_flag == 'inc':
                if valid[-1] < valid[-2] and valid[-2] < valid[-3] and valid[-3] < valid[-4] and valid[-4] < valid[-5]:
                    return True
            elif order_flag == 'des':
                if valid[-1] > valid[-2] and valid[-2] > valid[-3] and valid[-3] > valid[-4] and valid[-4] > valid[-5]:
                    return True
        return False

    # @profile
    def evaluate(self, Data_load, train_flag, topK=5):
        test_num_instances, test_num_loop = Data_load._check_batch_loop(flag=train_flag, batch_size=self.batch_size)
        for i in range(test_num_loop):
            x_ui, x_raw, x_node, y_list = Data_load._fetch_batch_data(flag=train_flag, batch_size=self.batch_size, current_loop=i)
            y_list.shape += (1,)

            feed_dict = {self.train_ui: x_ui, self.train_raw: x_raw, self.train_nodes: x_node, self.train_labels: y_list}
            y_batch_pred = self.sess.run((self.out), feed_dict=feed_dict)

            y_batch_pred = np.array(y_batch_pred.tolist())
            y_list = np.array(y_list.tolist())

            if i == 0:
                y_true = y_list.flatten()
                y_pred = y_batch_pred.flatten()
            else:
                y_true = np.concatenate([y_true, y_list.flatten()])
                y_pred = np.concatenate([y_pred, y_batch_pred.flatten()])

            # print('%s %d over %d' % (train_flag, i, train_num_loop))

        test_loss = log_loss(y_true, y_pred)

        if train_flag == 'train':
            test_hit, test_ndcg = 0., 0.
        else:
            test_hit, test_ndcg = eval_model_pro(y_true, y_pred, K=10, row_len=51)


        return test_hit, test_ndcg, test_loss

if __name__ == '__main__':
    # Data loading
    args = parse_args()

    node_generator = NodeGenerator(args)

    ui_M = node_generator.ui_M
    raw_M = node_generator.raw_M
    nodes_M = node_generator.nodes_M

    # Training
    t1 = time()
    model = NAR(ui_M, raw_M, nodes_M, args)

    final_result_str = model.train(node_generator)

    save_path = 'Output/'+args.out_dir+'/'+args.city+'_'+args.cate+'_'+args.data_type+'.txt'
    ensureDir(save_path)
    f = open(save_path, 'a')
    f.write("NAR: num_boost_round=%d, max_depth=%d, lr=%.4f, hidden_factor=%d, reg_embedding=%.4f, algorithm=%d, ui_flag=%d, bias_flag=%d, max_flag=%d, beta=%.4f %s\n"
              %(args.num_boost_round, args.max_depth, args.lr, args.hidden_factor, args.reg_embedding, args.algorithm, args.ui_flag, args.bias_flag, args.max_flag, args.beta, final_result_str))

    f.close()