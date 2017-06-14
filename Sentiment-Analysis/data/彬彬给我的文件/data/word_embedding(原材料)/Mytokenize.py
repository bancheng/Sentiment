# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:35:51 2016

@author: ASUS
"""

import nltk

f1 = open('data.txt','r')
allLines = f1.readlines()
f1.close()

f2 = open('token_data.txt','w')

for eachline in allLines:
    token_tmp = nltk.word_tokenize(eachline)
    for element in token_tmp:
        if element not in ['.',',','!','?','-','--','/','[',']','\'','(',')','<','>']:
            f2.write(element + ' ')
        
    f2.write('\n')

f2.close()
    

