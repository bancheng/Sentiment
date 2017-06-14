#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy

dataset_path='/home/tangdongge/下载/代码/测试代码/练习用文件'

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

n_dim = 300
#Initialize model and build vocab
imdb_w2v = Word2Vec(size=n_dim, min_count=10)
imdb_w2v.build_vocab(x_train)
#Train the model over train_reviews (this may take several minutes)
imdb_w2v.train(x_train)

#print((imdb_w2v["like"]).reshape((1,300)))


#Build word vector for training set by using the average value of all word vectors in the tweet, then scale
def buildWordVector(text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_train]) #np.concatenate把两个数组合并成一个
train_vecs = scale(train_vecs)

#Train word2vec on test tweets
imdb_w2v.train(x_test)

#Build test tweet vectors then scale
test_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_test])
test_vecs = scale(test_vecs)  #scale将给定数据进行标准化,减去其均值，并处以其方差。得到的结果是，对于每个属性/每列来说所有数据都聚集在0附近，方差为1。


lr = SGDClassifier(loss='log', penalty='l1')  #对于SGDClassifier，当loss=”log”时拟合成一个逻辑回归模型，
                                              # 当loss=”hinge”时拟合成一个线性支持向量机
lr.fit(train_vecs, y_train)  #加载数据

print 'Test Accuracy: %.2f'%lr.score(test_vecs, y_test)  #Returns the mean accuracy on the given test data and labels.
