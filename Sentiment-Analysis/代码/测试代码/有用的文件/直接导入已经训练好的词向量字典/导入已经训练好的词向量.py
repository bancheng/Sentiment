#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy

import numpy
import pickle as pkl

from collections import OrderedDict

import glob
import os
import shutil
from gensim.models.word2vec import Word2Vec

model = Word2Vec.load_word2vec_format('/home/tangdongge/下载/代码/测试代码/有用的文件/直接导入已经训练好的词向量字典/vectors.bin', binary=True) #C binary format
print model["like"].reshape(1,48)
