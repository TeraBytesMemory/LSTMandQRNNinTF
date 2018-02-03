# coding: utf-8
'''LSTMのクラス'''

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from abstract_neural_network import AbstractNeuralNetwork


class LSTM(AbstractNeuralNetwork):
    '''
    LSTMのクラス。ベクトル列を入力に、n_outクラスの分類を行う。

    Parameters
    -------------------
    n_in (int): 入力ベクトルのサイズ
    n_out (int): 出力クラスのサイズ
    n_hidden (int): 隠れ層のサイズ
    maxlen (int): ベクトル列の長さ
    load_path (str): 読み込みたいモデルのpath。読み込まない場合は空白。

    Note
    ---------
    self.x (Tensor): 入力のプレースホルダー
    self.t (Tensor): 正解のプレースホルダー
    self.n_batch (Tensor): バッチサイズのプレースホルダー

    self.y (Tensor): 予測値を出力するネットワーク
    self.loss (Tensor): 誤差を出力するネットワーク
    self.train_step(Tensor): 誤差伝搬を行う変数
    '''

    def __init__(self, n_in, n_out, n_hidden, maxlen=16, load_path=''):
        self.x = tf.placeholder(tf.float32, shape=[None, maxlen, n_in])
        self.t = tf.placeholder(tf.float32, shape=[None, n_out])
        self.n_batch = tf.placeholder(tf.int32)

        self.y = self._inference(self.x,n_hidden, n_out, self.n_batch, maxlen)

        self.loss = self._loss(self.y, self.t)
        self.train_step = self._train_step(self.loss)

        self.saver = tf.train.Saver()
        if load_path:
            self.sess = tf.InteractiveSession()
            self.saver.restore(self.sess, load_path)
        else:
            self.sess = tf.InteractiveSession()
            self.sess.run(tf.global_variables_initializer())

    def _inference(self, x, n_hidden, n_out, n_batch, maxlen=16):
        '''
        モデル全体の定義

        Parameters
        -------------------
        x (Tensor): 入力
        n_in (int): 入力ベクトルのサイズ
        n_hidden (int): 隠れ層のサイズ
        n_out (int): 出力クラスのサイズ
        n_batch (Tensor): バッチサイズ
        maxlen (int): ベクトル列の長さ

        Output
        -------------------
        Tensor: 予測値を出力するネットワーク
        '''
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        initial_state = cell.zero_state(n_batch, tf.float32)
        state = initial_state

        outputs = []
        with tf.variable_scope('LSTM'):
            for t in range(maxlen):
                if t > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(x[:, t, :], state)
                outputs.append(cell_output)
        output = outputs[-1]

        V = self._weight_variable([n_hidden, n_out])
        c = self._bias_variable([n_out])
        y = tf.matmul(output, V) + c
        y = tf.nn.softmax(y)

        return y

    def _loss(self, y, t):
        '''
        誤差関数の定義

        Parameters
        -------------------
        y (Tensor): 予測値
        t (Tensor): 正解の値

        Output
        -------------------
        Tensor: 誤差関数の出力
        '''
        cross_entropy = - tf.reduce_sum(t * tf.log(y+1e-10) + (1 - t) * tf.log(1 - y+1e-10))
        return cross_entropy

    def _train_step(self, loss):
        '''
        誤差伝搬の定義

        Parameters
        -------------------
        loss (Tensor): 誤差の値

        Output
        -------------------
        誤差伝搬に使う変数
        '''
        optimizer = tf.train.AdamOptimizer()
        train_step = optimizer.minimize(loss)
        return train_step

    def fit(self, x, y, n_epoch=200, batch_size=64):
        '''
        モデルの学習を行うメソッド

        Parameters
        -------------------
        x (list or np.array): 説明変数。行列の形は（入力データの数、ベクトル列の長さ、ベクトルの長さ）
        y (list or np.array): 目的変数（正解）。行列の形は（出力データの数、ベクトルの長さ）
        '''
        for epoch in range(1, n_epoch + 1):
            X_, Y_ = shuffle(x, y)
            sum_loss = 0
            for i in range(0, len(x), batch_size):
                batch_x = X_[i:i+batch_size]
                batch_y = Y_[i:i+batch_size]
                _, loss = self.sess.run([self.train_step, self.loss],
                                        feed_dict={self.x: batch_x, self.t: batch_y, self.n_batch: batch_size })
                sum_loss += loss

            sum_loss = sum_loss/(len(x)/batch_size)
            print("epoch {}, loss {}".format(epoch, sum_loss))

            if epoch %5 == 0 and epoch != 0:
                name = "lstm_model/model" + str(epoch) + ".ckpt"
                self.saver.save(self.sess, name)
                print("save ", name)

    def predict(self, x):
        '''
        予測を行うメソッド

        Parameters
        -------------------
        x (list or np.array): 説明変数。行列の形は（入力データの数、ベクトル列の長さ、ベクトルの長さ）

        Output
        -------------------
        y (np.array): 予測値。行列の形は（入力データの数、ベクトルの長さ）
        '''
        y = self.sess.run(self.y, feed_dict={self.x: x, self.n_batch: len(x) })
        return y

    def save(self, fname):
        '''
        モデルの保存を行うメソッド

        Parameters
        -------------------
        fname (str): モデルの名前
        '''
        self.saver.save(self.sess, fname)
        print("save", fname)
