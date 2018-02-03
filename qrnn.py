# coding: utf-8
'''QRNNのクラス'''

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from abstract_neural_network import AbstractNeuralNetwork


class QRNN(AbstractNeuralNetwork):
    '''
    QRNNのクラス。ベクトル列を入力に、n_outクラスの分類を行う。

    Parameters
    -------------------
    n_in (int): 入力ベクトルのサイズ
    n_out (int): 出力クラスのサイズ
    n_hidden (int): 隠れ層のサイズ
    n_layers (int): 畳み込み層の数（未実装）
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

    def __init__(self, n_in, n_out, n_hidden, n_layers=1, maxlen=16, load_path=''):
        self.x = tf.placeholder(tf.float32, shape=[None, maxlen, n_in])
        self.t = tf.placeholder(tf.float32, shape=[None, n_out])
        self.n_batch = tf.placeholder(tf.int32)

        self.y = self._inference(self.x, n_in, n_hidden, n_out, maxlen, self.n_batch, n_layers)

        self.loss = self._loss(self.y, self.t)
        self.train_step = self._train_step(self.loss)

        self.saver = tf.train.Saver()
        if load_path:
            self.sess = tf.InteractiveSession()
            self.saver.restore(self.sess, load_path)
        else:
            self.sess = tf.InteractiveSession()
            self.sess.run(tf.global_variables_initializer())

    def _inference(self, x, n_in, n_hidden, n_out, maxlen, n_batch, n_layers):
        '''
        モデル全体の定義

        Parameters
        -------------------
        x (Tensor): 入力
        n_in (int): 入力ベクトルのサイズ
        n_hidden (int): 隠れ層のサイズ
        n_out (int): 出力クラスのサイズ
        maxlen (int): ベクトル列の長さ
        n_batch (Tensor): バッチサイズ
        n_layers (int): 畳み込み層の数（未実装）

        Output
        -------------------
        Tensor: 予測値を出力するネットワーク

        TODO
        -------------------
        畳み込み層の数を増やす
        '''
        state = tf.zeros([self.n_batch, n_hidden], dtype=tf.float32)

        h = x
        #for i in range(n_layers): # TODO: Future
        f, z, o = self._conv(h, n_in, n_hidden, name='conv_{}'.format(0))
        h, state = self._fo_pool(f, z, o, state, maxlen, name='fo_pool_{}'.format(0))

        y = self._linear(h, n_hidden, n_out)
        y = tf.nn.softmax(y)

        return y


    def _conv(self, x, n_in, n_hidden, name='conv'):
        '''
        畳み込み層の定義

        Parameters
        -------------------
        n_in (int): 入力ベクトルのサイズ
        n_hidden (int): 隠れ層のサイズ
        name (str): 名前空間の定義

        Output
        -------------------
        tuple(Tensor, Tensor, Tensor): 畳み込み層の出力（それぞれ忘却変数、隠れ層の変数、隠れ層の変数）
        '''
        with tf.variable_scope(name):
            w = self._weight_variable([3, n_in, n_hidden * 3])
            conv = tf.nn.conv1d(x, w, stride=1, padding="SAME", data_format="NHWC")

            conv = tf.transpose(conv, [1, 0, 2])
            f, z, o = tf.split(conv, 3, 2)

        return f, z, o

    def _fo_pool(self, f, z, o, c, maxlen, name='fo_pool'):
        '''
        プーリング層（fo-pool）の定義

        Parameters
        -------------------
        f (Tensor): 忘却変数
        z (Tensor): 隠れ層の変数
        o (Tensor): 隠れ層の変数
        c (Tensor): 状態を示す変数

        Output
        -------------------
        tuple(Tensor, Tensor): プーリング層の出力（それぞれ隠れ層の出力、状態を示す変数）
        '''
        with tf.variable_scope(name, reuse=True):
            for i in range(maxlen):
                f_ = tf.sigmoid(f[i])
                z_ = tf.tanh(z[i])
                o_ = tf.sigmoid(o[i])
                c = f_ * c + 1 - f_ * z_
                h = o_ * c
        return h, c

    def _linear(self, x, n_hidden, n_out):
        '''
        線形関数の定義

        Parameters
        -------------------
        n_hidden (int): 隠れ層のユニット数
        n_out (int): 出力層のユニット数

        Output
        -------------------
        Tensor: 線形関数の出力
        '''
        with tf.variable_scope('linear'):
            w = self._weight_variable([n_hidden, n_out])
            b = self._bias_variable([n_out])
            y = tf.matmul(x, w) + b
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
                                        feed_dict={self.x: batch_x, self.t: batch_y , self.n_batch: batch_size})
                sum_loss += loss

            sum_loss = sum_loss/(len(x)/batch_size)
            print("epoch {}, loss {}".format(epoch, sum_loss))

            if epoch %5 == 0 and epoch != 0:
                name = "qrnn_model/model" + str(epoch) + ".ckpt"
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
