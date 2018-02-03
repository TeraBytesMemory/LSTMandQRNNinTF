# coding: utf-8
'''行から文字ごとへ分割するクラスのテンプレート'''
from abc import ABCMeta, abstractmethod
import tensorflow as tf

class AbstractNeuralNetwork(metaclass=ABCMeta):
    
    
    def _weight_variable(self, shape):
        """
        モデルの重みを定義するmethod
        
        Parameters
        -------------------
        shape(list): 重みのサイズを渡す配列
        
        Output
        -------------------
        tf.Variable: ネットワークの重み
        """
        initial = tf.truncated_normal(shape, stddev=0.01)
        
        return(tf.Variable(initial))

    def _bias_variable(self, shape):
        """
        モデルのバイアスを定義するmethod
        
        Parameters
        -------------------
        shape(list): バイアスのサイズを渡す配列
        
        Output
        -------------------
        tf.Variable: ネットワークのバイアス
        """
        initial = tf.zeros(shape)
        
        return(tf.Variable(initial))
        
    @abstractmethod
    def _inference(self):
        '''
        モデル全体を定義するmethod
        '''
        pass
    
    @abstractmethod
    def _loss(self, y_, y):
        """
        lossを計算するmethod
        """
        pass
    
    def _accuracy(self, y_, y):
        """
        accuracyを計算するmethod
        
        Parameters
        -------------------
        y_(tensor): 正解ラベル
        y(tensor): 予測ラベル
        
        Output
        -------------------
        tensor: accuracyの値
        """
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        return accuracy
    
    @abstractmethod
    def _train_step(self):
        """
        最適化関数を定義してトレーニングを行うmethod
        """
        pass
    
    @abstractmethod
    def fit(self):
        '''
        モデルの学習を行う method
        '''
        pass
    
    @abstractmethod
    def predict(self):
        '''
        学習したモデルで予測する method
        '''
        pass