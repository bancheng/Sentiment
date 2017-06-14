#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy

dataset_path='/home/tangdongge/下载/代码/测试代码/练习用文件'
import cPickle as pkl
import numpy as np
#model = Word2Vec.load_word2vec_format('/home/tangdongge/下载/代码/测试代码/练习用文件/0_3.txt', binary=False)
from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import scale
from sklearn.linear_model import SGDClassifier

f = open("/home/tangdongge/data/强行复制/f2","a")
fff = open("/home/tangdongge/data/强行复制/f1","a")
with open('/home/tangdongge/下载/代码/测试代码/练习用文件/data_review_pos.txt', 'r') as infile:
    pos_tweets = infile.readlines()   #   把文本转化成列表

with open('/home/tangdongge/下载/代码/测试代码/练习用文件/data_review_neg.txt', 'r') as infile:
    neg_tweets = infile.readlines()

#use 1 for positive sentiment, 0 for negative
y = np.concatenate((np.ones(len(pos_tweets)), np.zeros(len(neg_tweets))))

x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_tweets, neg_tweets)), y, test_size=0.2)
#cross_validation.train_test_split  从样本中随机的按比例选取train data和test data。 test_size是测试集样本占比。
# 如果是整数的话就是样本的数量。random_state是随机数的种子。不同的种子会造成不同的随机采样结果。相同的种子采样结果相同。

#Do some very minor text preprocessing
def cleanText(corpus):
    corpus = [z.lower().replace('\n','').split() for z in corpus]
    return corpus

x_train = cleanText(x_train)
x_test = cleanText(x_test)
f = open('imdb.pkl', 'wb')
pkl.dump((x_train, y_train), f, -1)
pkl.dump((x_test, y_test), f, -1)
f.close()
