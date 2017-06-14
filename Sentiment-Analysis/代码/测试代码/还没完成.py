#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy

#用MLP训练词袋
import numpy as np
import pickle as pkl

from collections import OrderedDict
import glob
from subprocess import Popen, PIPE
import os
import re
import nltk
from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import scale
from sklearn.linear_model import SGDClassifier
import theano
import theano.tensor as T
#把文章中的标点符号去掉，改成小写
def _tokenize(data):
    os.chdir(data)
    with open("token_data_neg.txt",'r') as f1:
        context1=f1.read()
        context1=re.sub(r"[, .' / < > ! ? \ : \" + = ( ) ^ * # ]"," ",context1)
        context1=re.sub(r"\d"," ",context1)
        context1=context1.lower()
        with open("_token_data_neg.txt",'w') as f3:
            f3.write(context1)
    with open("token_data_pos.txt",'r') as f2:
        context2=f2.read()
        context2=re.sub(r"[, .' / < > ! ? \ : \" + = ( ) ^ * # ]"," ",context2)
        context2=re.sub(r"\d"," ",context2)
        context2=context2.lower()
        with open("_token_data_pos.txt",'w') as f4:
            f4.write(context2)
#建立一个包含所有单词的词典，key是单词，value是单词对应的位置
def buildlib(data):
    os.chdir(data)
    with open("_token_data_neg.txt",'r') as f1:
        context5=f1.read()
    with open("_token_data_pos.txt",'r') as f2:
        context6=f2.read()
    context=context5+context6
    print"building a lib"
    _dict=dict()
    i=0
    ss=context.split()
    for word in ss:
        if word not in _dict:
            i=i+1
            _dict[word]=i-1
    return _dict
#对正集建立词带
def _vectneg(data,_dict):
    os.chdir(data)
    dict_size=len(_dict)
    vect_sentence=[]
    with open("_token_data_neg.txt",'r') as f7:
        while True:
           context7=f7.readline()
           if not context7:
               break
           ss=context7.split()
           vect_sentence1=np.zeros(dict_size)
           for word in ss:
               vect_sentence1[_dict[word]]=1
           vect_sentence.append(vect_sentence1)
    return vect_sentence
#对负集建立词带
def _vectpos(data,_dict):
    os.chdir(data)
    dict_size=len(_dict)
    vect_sentence=[]
    with open("_token_data_pos.txt",'r') as f7:
        while True:
           context7=f7.readline()
           if not context7:
               break
           ss=context7.split()
           vect_sentence1=np.zeros(dict_size)
           for word in ss:
               vect_sentence1[_dict[word]]=1
           vect_sentence.append(vect_sentence1)
    return vect_sentence
#给正负集加标签
def _label(vect_sentence1,vect_sentence2):
    train_x=vect_sentence1+vect_sentence2
    train_y = [1] * len(vect_sentence2) + [0] * len(vect_sentence1)
    return train_x,train_y

def _division(train_x,train_y):
    x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2,random_state=0)
    x_train,x_valid,y_train,y_valid=train_test_split(train_x,train_y,test_size=0.4,random_state=0)
    return x_valid,x_train,x_test,y_valid,y_train,y_test
if __name__ == '__main__':
    data = "/home/tangdongge/下载/练习用文件"
    _dict = buildlib(data)
    vect_sentence1 = _vectneg(data,_dict)
    vect_sentence2 = _vectpos(data,_dict)
    train_x,train_y = _label(vect_sentence1,vect_sentence2)
    x_valid,x_train,x_test,y_valid,y_train,y_test=_division(train_x,train_y)
    os.chdir("/home/tangdongge/下载/代码/测试代码")
    f = open('imdb.pkl', 'wb')
    pkl.dump((x_valid,y_valid),f,-1)
    pkl.dump((x_train, y_train), f, -1)
    pkl.dump((x_test, y_test), f, -1)
    f.close()
















