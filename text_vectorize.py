# coding: utf-8
'''テキスト文書を文字や単語ごとにonehotベクトル化するクラス'''

import numpy as np


class TextVectorize(object):
    '''
    テキスト文書を文字や単語ごとにonehotベクトル化するクラス

    Parameters
    -------------------
    text (str or list(str)): テキスト文書もしくは分かち書きしたリスト型のテキスト文書。
    文字にわけたい時はstrで、単語に分けたい時はあらかじめ形態素解析してリストに分けること。

    Note
    ---------
    self.char_to_id (dict): 単語からIDへのマッピング
    self.id_to_char (dict): IDから単語へのマッピング
    '''
    def __init__(self, text):
        self.text_set = sorted(list(set(text)))
        self.char_to_id = { c: i  for i, c in enumerate(self.text_set) }
        self.id_to_char = { i: c  for i,c  in enumerate(self.text_set) }

    def one_hot(self, text):
        '''
        テキストを文字や単語ごとのonehotベクトル化するメソッド

        Parameters
        -------------------
        text (str or list(str)): テキスト文書もしくは分かち書きしたリスト型のテキスト文書。
        文字にわけたい時はstrで、単語に分けたい時はあらかじめ形態素解析してリストに分けること。

        Output
        -------------------
        list(np.array): 文字や単語ごとのonehotベクトルのリスト
        '''
        vectors = []
        for char in text:
            new_vector = np.zeros(len(self.text_set))
            i = self.char_to_id[char]
            new_vector[i] = 1
            vectors.append(new_vector)
        return vectors

    def pad(self, vectors, maxlen):
        '''
        ベクトル列の頭にゼロベクトルを追加し、ベクトル列の長さをmaxlenにする。

        Parameters
        -------------------
        vectors (list(np.array)): 文字や単語ごとのonehotベクトルのリスト
        maxlen(int): ベクトル列の長さ

        output
        -------------------
        vectors (list(np.array)): 文字や単語ごとのonehotベクトルのリスト
        '''
        return [np.zeros(len(self.text_set)) for _ in range(maxlen - len(vectors))] + vectors
