# coding: utf-8
import os
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models import Word2Vec
import numpy as np


class LdaWord2VecModel:
    
    def __init__(self, corpus, w2v_size=100, topics=100, w2v_path='', lda_path=''):
        """initialize LdaWord2VecModel
        initialize and train the LdaWord2VecModel according to the args.
        
        Args:
            corpus: the corpus used to train, a sequence of sequence of words.
            w2v_size: size of word vector, 100 by default.
            topics: num_topics of LDA model, 100 by default.
            w2v_path: the path to load or save word vector model, '' by default, which means not loading or saving.
            lda_path: the path to load or save LDA model, '' by default, which means not loading or saving.
        """
        # 保留类的初始化变量
        self.topics = topics
        self.w2v_size = w2v_size
        
        # 训练或载入词向量模型
        if w2v_path != '': 
            if os.path.exists(w2v_path):
                self.w2v_model = Word2Vec.load(w2v_path)
            else:
                self.w2v_model = Word2Vec(corpus, size=w2v_size)
                self.w2v_model.save(w2v_path)
        else:
            self.w2v_model = Word2Vec(corpus, size=w2v_size)
        
        # 训练或载入LDA模型
        if lda_path != '':
            if os.path.exists(lda_path):
                self.lda_model = LdaModel.load(lda_path)
            else:
                word_dict = Dictionary(corpus)
                bow_corpus = self.BowCorpus(word_dict, corpus)
                self.lda_model = LdaModel(bow_corpus, id2word=word_dict, num_topics=topics)
                self.lda_model.save(lda_path)
        else:
            word_dict = Dictionary(corpus)
            bow_corpus = self.BowCorpus(word_dict, corpus)
            self.lda_model = LdaModel(bow_corpus, id2word=word_dict, num_topics=topics)
        
        # 计算主题向量
        topic_bow = self.lda_model.show_topics(num_topics=-1)
        self.topic_vecs = []
        for topic in topic_bow:
            vec = np.zeros(w2v_size, dtype=float)
            for word_tuple in topic[1].split(' + '):
                weight, word = word_tuple.split('*')
                if word[1:-1] in self.w2v_model.wv:
                    vec += self.w2v_model.wv[word[1:-1]]*float(weight)
            self.topic_vecs.append(vec)
    
    def get_topics(self, topics=10, words=10):
        return self.lda_model.show_topics(num_topics=topics, num_words=words)
        
    def get_topic_vecs(self):
        return self.topic_vecs

    def get_word_vecs(self):
        return self.w2v_model.wv
        
    def predict(self, doc):
        # 计算文档向量
        doc_vec = np.array(w2v_size, dtype-float)
        for sent in doc:
            for word in sent:
                if word in self.w2v_model.wv: doc_vec += self.w2v_model.wv[word]
        
        # 寻找余弦相似度最大的主题向量
        topic = -1        
        cos_max = 0
        for i in range(len(self.topic_vecs)):
            cos = np.dot(doc_vec, self.topic_vecs[i]) / np.sqrt(sum(doc_vec**2) * sum(self.topic_vec[i]**2))
            if cos >= cos_max:
                topic = i
                cos_max = cos
                
        return topic
    
    class BowCorpus:
        
        def __init__(self, word_dict, corpus):
            self.word_dict = word_dict
            self.corpus = corpus
        
        def __iter__(self):
            for doc in self.corpus: yield self.word_dict.doc2bow(doc)
            