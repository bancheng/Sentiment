#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy

dataset_path='/home/tangdongge/下载/代码/LSTM Networks for Sentiment Analysis/刘竞爽代码/aclImdb/'

import numpy
import pickle as pkl

from collections import OrderedDict

import glob
import os

from subprocess import Popen, PIPE

# tokenizer.perl is from Moses: https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer
tokenizer_cmd = ['./tokenizer.perl', '-l', 'en', '-q', '-']


def tokenize(sentences): #不知道这个函数是干嘛用的？

    print ('Tokenizing..',)
    text = "\n".join(sentences)  # 方法用于将序列中的元素以指定的字符连接生成一个新的字符串。
    tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)  #
    tok_text, _ = tokenizer.communicate(text)  #Popen.communicate(input=None)与子进程进行交互。
#  向stdin发送数据，或从stdout和stderr中读取数据。
#可选参数input指定发送到子进程的参数。Communicate()返回一个元组：(stdoutdata, stderrdata)。
# 注意：如果希望通过进程的stdin向其发送数据，在创建Popen对象的时候，参数stdin必须被设置为PIPE。
#同样，如果希望从stdout和stderr获取数据，必须将stdout和stderr设置为PIPE
    toks = tok_text.split('\n')[:-1]   #str.split(str="", num=string.count(str))[n], 通过指定分隔符对字符串进行切片
                                       #  str -- 分隔符，默认为空格。num -- 分割次数。[n]表示选取第n个分片

    print ('Done')

    return toks


def build_dict(path):
    sentences = []
    currdir = os.getcwd()   #返回所运行脚本的目录
    os.chdir('%s/pos/' % path) #改变当前工作目录到指定的路径
    for ff in glob.glob("*.txt"):   #glob.glob  返回所有匹配的文件路径列表
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())  #s.strip(rm)删除s字符串中开头、结尾处，位于 rm删除序列的字符,当rm为空时，
                                                    # 默认删除空白符（包括'\n', '\r',  '\t',  ' ')
    os.chdir('%s/neg/' % path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir(currdir)

    sentences = tokenize(sentences)  #不知道这个是干什么的？

    print ('Building dictionary..',)
    wordcount = dict()
    for ss in sentences:
        words = ss.strip().lower().split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1  #记录了每个单词出现的次数

    counts = wordcount.values()
    keys = wordcount.keys()

    sorted_idx = numpy.argsort(counts)[::-1] #numpy.argsort数组值从小到大的索引值,第一个数的索引始终是0；-1表示从大到小

    worddict = dict()

    for idx, ss in enumerate(sorted_idx): #enumerate函数用于遍历序列中的元素以及它们的下标
        worddict[keys[ss]] = idx+2  # leave 0 and 1 (UNK)

    print (numpy.sum(counts), ' total words ', len(keys), ' unique words')

    return worddict


def grab_data(path, dictionary):
    sentences = []
    currdir = os.getcwd()
    os.chdir(path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir(currdir)
    sentences = tokenize(sentences) #又是这个函数

    seqs = [None] * len(sentences)
    for idx, ss in enumerate(sentences):
        words = ss.strip().lower().split()
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]

    return seqs


def main():
    # Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/
    path = dataset_path
    dictionary = build_dict(os.path.join(path, 'train')) #os.path.join 把多个路径合在一起

    train_x_pos = grab_data(path+'train/pos', dictionary)
    train_x_neg = grab_data(path+'train/neg', dictionary)
    train_x = train_x_pos + train_x_neg
    train_y = [1] * len(train_x_pos) + [0] * len(train_x_neg)

    test_x_pos = grab_data(path+'test/pos', dictionary)
    test_x_neg = grab_data(path+'test/neg', dictionary)
    test_x = test_x_pos + test_x_neg
    test_y = [1] * len(test_x_pos) + [0] * len(test_x_neg)

    f = open('imdb.pkl', 'wb')
    pkl.dump((train_x, train_y), f, -1)
    pkl.dump((test_x, test_y), f, -1)
    f.close()

    f = open('imdb.dict.pkl', 'wb')
    pkl.dump(dictionary, f, -1)  #pickle.dump(obj, file, [,protocol] 注解：将对象obj保存到文件file中去
    f.close()

if __name__ == '__main__':
    main()
