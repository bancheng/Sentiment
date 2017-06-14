#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy
#注意readline()每次只能读一行，所以要用永真循环。read可以一次读取多行，但也是按照一行一行的顺序读的。尽量避免使用readlines
import numpy
import pickle as pkl

from collections import OrderedDict

import glob
import os
def _combine(path1,path2):
    with open(path2,"a") as fff:
       os.chdir(path1)
       for ff in glob.glob("*.txt"):
           with open(ff,'r') as f:
               while True:
                    contexte = f.readline()
                    if contexte:
                        fff.write(contexte.strip())
                    else:
                        break
           fff.write('\n')
    return path2
if __name__ == '__main__':
    _combine("/home/tangdongge/data/review_polarity 2.0/txt_sentoken/neg","/home/tangdongge/下载/代码/测试代码/练习用文件/data_review_neg.txt")
    _combine("/home/tangdongge/data/review_polarity 2.0/txt_sentoken/pos","/home/tangdongge/下载/代码/测试代码/练习用文件/data_review_pos.txt")

