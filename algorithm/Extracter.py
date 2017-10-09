# coding: utf-8
import os
import re
from gensim.models import Word2Vec
import jieba
import numpy as np
from sklearn.decomposition import PCA


class Extracter:

    def __init__(self, wv={}):
        self.wv = wv

    def normalize(self, doc):
        # python的re模块无法处理中文标点，故处理之
        return  doc.replace('。', '.')\
            .replace('？', '?')\
            .replace('！', '!')\
            .replace('：', ':')\
            .replace('，', ',')\
            .replace('；', ';')

    def extract(self, corpus, stopwords=[]):
        if len(self.wv) == 0: self.wv = Word2Vec(corpus).wv

        result = []

        for doc in corpus:
            # 计算每个句子的句向量
            vecs = []
            for sent in re.split(r'[.!?:;]', normalize(doc)):
                wordVecs = [self.wv[word] for word in jieba.cut(sent) if word in self.model]
                if len(wordVecs) != 0: vecs.append(sum(wordVecs)/len(wordVecs))
            vecs = np.array(vecs)

            # 标记转折句的位置
            dis = [np.sqrt(sum((vecs[i] - vecs[i+1])**2)) for i in range(len(vecs)-1)]
            half = (max(dis) - min(dis)) / 2
            flag = [0] + [i+1 for i in range(len(dis)) if dis[i] > min(dis) + half and i > 1]

            # 将内容分段
            result.append([doc[flag[i]:flag[i+1]] for i in range(len(flag)-1)] + doc[flag[-1]:-1])

        return result
