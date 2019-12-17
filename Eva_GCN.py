from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
import pickle

import numpy as np
import tensorflow as tf
from utilz import get_shuffled_data
# from src.utils import get_train_ops

class evaluator(object):

    def __init__(
        self,
        num_of_nodes,
        whole_channels,
        num_of_branches):

        self.gcn_num_layers = 2
        self.gcn_size = num_of_nodes
        self.name = "evaluator"
        self.num_of_nodes = num_of_nodes
        self.num_of_branches = num_of_branches
        self.whole_channels = whole_channels

        self._create_params()
        self._build_train()


    def _create_params(self):

        initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        with tf.variable_scope(self.name, initializer=initializer):

            with tf.variable_scope("gcn"):
                self.w_gcn = []
                with tf.variable_scope("gcn_layer_{}".format(1)):
                    w = tf.get_variable("w", [6, 12])
                    self.w_gcn.append(w)
                with tf.variable_scope("gcn_layer_{}".format(2)):
                    w = tf.get_variable("w", [12, 24])
                    self.w_gcn.append(w)

            with tf.variable_scope("nn",initializer=initializer):
                self.w_nn = []
                self.b_nn = []
                with tf.variable_scope("nn_layer_{}".format(1)):
                    w = tf.get_variable("w", [288, 64])
                    b = tf.get_variable("b",[1,64])
                    self.w_nn.append(w)
                    self.b_nn.append(b)
                with tf.variable_scope("nn_layer_{}".format(2)):
                    w = tf.get_variable("w", [64, 32])
                    b = tf.get_variable("b",[1,32])
                    self.w_nn.append(w)
                    self.b_nn.append(b)
                with tf.variable_scope("nn_layer_{}".format(3)):
                    w = tf.get_variable("w", [32, 16])
                    b = tf.get_variable("b",[1,16])
                    self.w_nn.append(w)
                    self.b_nn.append(b)
                with tf.variable_scope("nn_layer_{}".format(4)):
                    w = tf.get_variable("w", [16, 1])
                    b = tf.get_variable("b",[1,1])
                    self.w_nn.append(w)
                    self.b_nn.append(b)
                  

    def _build_train(self):

        supports = tf.placeholder(dtype=tf.float32,shape=[None,12,12],name="support")
        features = tf.placeholder(dtype=tf.float32,shape=[None,12,6],name = "feature")
        labels = tf.placeholder(dtype=tf.float32,shape=[None,1],name="label")

        supports_1 = tf.matmul(supports,features) # None 12 6
        weight_1 = self.w_gcn[0] # 6,12
        result_1 = tf.nn.relu(tf.matmul(tf.reshape(supports_1,[-1,6]),weight_1))


        support_2 = tf.matmul(supports,tf.reshape(result_1,[-1,12,12]))
        weight_2 = self.w_gcn[1]
        result_2 = tf.nn.relu(tf.matmul(tf.reshape(support_2,[-1,12]),weight_2))
        result_2 = tf.reshape(result_2,[-1,12,24])

        flatten = tf.layers.Flatten()(result_2)
        dense_1 = tf.nn.relu(tf.matmul(flatten,self.w_nn[0]) + self.b_nn[0])
        dense_1 = tf.nn.relu(tf.matmul(dense_1,self.w_nn[1]) + self.b_nn[1])
        dense_1 = tf.nn.relu(tf.matmul(dense_1,self.w_nn[2]) + self.b_nn[2])
        accuracy = tf.nn.sigmoid(tf.matmul(dense_1,self.w_nn[3]) + self.b_nn[3],name="accuracy")

        loss = tf.losses.mean_squared_error(labels=labels,predictions=accuracy)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train = optimizer.minimize(loss)
        saver = tf.train.Saver([self.w_nn[0],self.w_nn[1],self.w_nn[2],self.w_nn[3],
                                self.b_nn[0],self.b_nn[1],self.b_nn[2],self.b_nn[3],
                                self.w_gcn[0],self.w_gcn[1]])

        self.train_op = train
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            support,feature,label,onehot_feature = get_shuffled_data()
            data_size = 100
            losses = []
            val_losses = []
            test_losses = []

            epoch = 100
            batch_size = 32
            batch_num = int(data_size/batch_size)
            for i in range(epoch):
                for j in range(batch_num):
                    low = batch_size*j
                    high = batch_size*(j+1)
                    _,losss = sess.run((train,loss),
                            feed_dict={supports:support[low:high],features:onehot_feature[low:high],labels:label[low:high]})

                val_loss = sess.run(loss,feed_dict={supports:support[data_size:int(data_size*1.2)],
                                                    features:onehot_feature[data_size:int(data_size*1.2)],
                                                    labels:label[data_size:int(data_size*1.2)]})

                test_loss = sess.run(loss,feed_dict={supports:support[int(data_size*1.4):int(data_size*1.8)],
                                                    features:onehot_feature[int(data_size*1.4):int(data_size*1.8)],
                                                    labels:label[int(data_size*1.4):int(data_size*1.8)]})

                losses.append(losss)
                val_losses.append(val_loss)
                test_losses.append(test_loss)
                print("epoch:",i,"mean_square_loss:",losss,"validation loss:", val_loss,"test_loss:",test_loss)
            saver.save(sess,'./checkpoint_dir/evaluator')

            w_gcn = sess.run(self.w_gcn)
            w_nn = sess.run(self.w_nn)
            b_nn = sess.run(self.b_nn)
            with open("Data/weight_gcn","wb") as f:
                weights = {"w_gcn":w_gcn,"w_nn":w_nn,"b_nn":b_nn}
                pickle.dump(weights,f)
            

    def _model(self,supports,features):

        # supports = tf.convert_to_tensor(supports,dtype=tf.float32)
        # features = tf.convert_to_tensor(features,dtype=tf.float32)

        # supports_1 = tf.matmul(supports,features)
        # weight_1 = self.w_gcn[0]
        # result_1 = tf.nn.relu(tf.matmul(supports_1,weight_1))

        # support_2 = tf.matmul(supports,result_1)
        # weight_2 = self.w_gcn[1]
        # result_2 = tf.nn.relu(tf.matmul(support_2,weight_2))

        # result_2 = tf.reshape(result_2,[-1,12,24])

        # flatten = tf.layers.Flatten()(result_2)

        # dense_1 = tf.nn.relu(tf.matmul(flatten,self.w_nn[0]) + self.b_nn[0])
        # dense_1 = tf.nn.relu(tf.matmul(dense_1,self.w_nn[1]) + self.b_nn[1])
        # dense_1 = tf.nn.relu(tf.matmul(dense_1,self.w_nn[2]) + self.b_nn[2])
        # accuracy = tf.nn.sigmoid(tf.matmul(dense_1,self.w_nn[3]) + self.b_nn[3])
        supports = np.reshape(supports,(-1,12,12))
        features = np.reshape(features,(-1,12,6))

        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph('./checkpoint_dir/evaluator.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))

            graph = tf.get_default_graph()
            support = graph.get_tensor_by_name("support:0")
            feature = graph.get_tensor_by_name("feature:0")
            feed_dict = {support:supports,feature:features}
            predict = graph.get_tensor_by_name("accuracy:0")

            accuracy = sess.run(predict,feed_dict)

        return accuracy      

    def evaluate(self,arc):

        submodel = arc
        A = np.zeros(shape = (self.num_of_nodes,self.num_of_nodes))
        D = np.zeros(shape = (self.num_of_nodes,self.num_of_nodes))
        feature = np.zeros(shape=(self.num_of_nodes,6))
        for i in range(self.num_of_nodes):
            layer = submodel[i][1:]
            feature[i][submodel[i][0]] = 1
            for j in range(len(layer)):
                A[i][j] = layer[j]
        A = A + np.transpose(A)
        for i in range(self.num_of_nodes):
            D[i][i] = sum(A[i])
        supports = np.eye(self.num_of_nodes,self.num_of_nodes) - np.matmul(np.matmul(np.sqrt(D),A),np.sqrt(D))
        # supports = np.matmul(supports,feature)

        eval_acc = self._model(supports,feature)
        return eval_acc[0][0]
        # return 0


class offline_evaluator(object):

    def __init__(self,
        num_of_nodes,
        whole_channels,
        num_of_branches):

        self.gcn_num_layers = 2
        self.gcn_size = num_of_nodes
        self.name = "evaluator"
        self.num_of_nodes = num_of_nodes
        self.num_of_branches = num_of_branches
        self.whole_channels = whole_channels

        self.w_gcn = []
        self.w_nn = []
        self.b_nn = []

        self._load_weighs()

    def sigmoid(self,x):
        s = 1 / (1 + np.exp(-x))
        return s

    def relu(self,x):
        s = np.where(x < 0, 0, x)
        return s

    def _load_weighs(self):
        with open("Data/weight_gcn","rb") as f:
            weights = pickle.load(f)
            self.w_gcn = weights["w_gcn"]
            self.w_nn = weights["w_nn"]
            self.b_nn = weights["b_nn"]

    def _build_model_by_np(self,supports,features):


        supports_1 = np.matmul(supports,features) # None 12 6
        weight_1 = self.w_gcn[0] # 6,12
        result_1 = self.relu(np.matmul(np.reshape(supports_1,[-1,6]),weight_1))


        support_2 = np.matmul(supports,np.reshape(result_1,[-1,12,12]))
        weight_2 = self.w_gcn[1]
        result_2 = self.relu(np.matmul(np.reshape(support_2,[-1,12]),weight_2))
        result_2 = np.reshape(result_2,[-1,12,24])

        flatten = result_2.flatten()
        dense_1 = self.relu(np.matmul(flatten,self.w_nn[0]) + self.b_nn[0])
        dense_2 = self.relu(np.matmul(dense_1,self.w_nn[1]) + self.b_nn[1])
        dense_3 = self.relu(np.matmul(dense_2,self.w_nn[2]) + self.b_nn[2])
        accuracy = self.sigmoid(np.matmul(dense_3,self.w_nn[3]) + self.b_nn[3])

        return accuracy

    def evaluate(self,arc):
        
        submodel = arc
        A = np.zeros(shape = (self.num_of_nodes,self.num_of_nodes))
        D = np.zeros(shape = (self.num_of_nodes,self.num_of_nodes))
        feature = np.zeros(shape=(self.num_of_nodes,6))
        for i in range(self.num_of_nodes):
            layer = submodel[i][1:]
            feature[i][submodel[i][0]] = 1
            for j in range(len(layer)):
                A[i][j] = layer[j]
        A = A + np.transpose(A)
        for i in range(self.num_of_nodes):
            D[i][i] = sum(A[i])
        supports = np.eye(self.num_of_nodes,self.num_of_nodes) - np.matmul(np.matmul(np.sqrt(D),A),np.sqrt(D))
        # supports = np.matmul(supports,feature)

        eval_acc = self._build_model_by_np(supports,feature)
        return eval_acc[0][0]


if __name__ == "__main__":
    eval_model = evaluator(12,True,8)
    offline_eval_model = offline_evaluator(12,True,8) 
    arc = [
        [0],
        [0,0],
        [2,0,0],
        [5,1,1,0],
        [1,1,0,0,0],
        [0,1,1,1,0,0],
        [4,0,0,1,1,1,0],
        [4,1,0,0,1,1,0,0],
        [5,1,1,0,1,0,0,1,0],
        [2,0,1,1,0,0,1,0,1,1],
        [2,0,1,1,0,0,1,0,1,1,1],
        [2,0,1,1,0,0,1,0,1,1,1,0]]

    print(eval_model.evaluate(arc))
    print(offline_eval_model.evaluate(arc))

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())

    #     # a = sess.run(eval_model.evaluate(arc))
    #     print(sess.run(eval_model.w_gcn[0]))
        

    #     eval_model._build_train()

    #     print(sess.run(eval_model.w_gcn[0]))
    #     # a = sess.run(eval_model.evaluate(arc))
    #     # print(a)