#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy


import numpy
import pickle as pkl

from collections import OrderedDict

import glob
import os
import re
import nltk
context1="you are good"

m=nltk.sent_tokenize(context1)
print m
print m[0]
#print m[1]
print m[0][1]
words=[]
for sent in m:
    words.append(nltk.word_tokenize(sent))
print words
print words[0]
#print words[1]
print words[0][1]

