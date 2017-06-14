import numpy as np
import scipy.io as sio

f1 = open('word_list_unstem4.txt','r')
words_str = f1.readline()
words_list = words_str.split()
feature_length = len(words_list)
f1.close()

## 用10000个词汇来表示25000条负极的影评句子
f2 = open('token_data_neg.txt','r')
matrix_doc_neg = np.zeros((25000,feature_length))
for line in range(25000):
    sentance = f2.readline()
    sentance_list = sentance.split()
    for i in range(len(sentance_list)):
        for j in range(len(words_list)):
            if sentance_list[i] == words_list[j]:
                matrix_doc_neg[line][j] = 1
                break

f2.close()

## 用XX个词汇来表示25000条正极的影评句子
f3 = open('token_data_pos.txt','r')
matrix_doc_pos = np.zeros((25000,feature_length))
for line in range(25000):
    sentance = f3.readline()
    sentance_list = sentance.split()
    for i in range(len(sentance_list)):
        for j in range(len(words_list)):
            if sentance_list[i] == words_list[j]:
                matrix_doc_pos[line][j] = 1
                break

f3.close()

## partioning 分training set & test set
# 25000 neg & 25000 pos
# 按7:3分割，17500*2 : 7500*2
k1 = np.random.permutation(25000)
k2 = np.random.permutation(25000)
train_neg = np.zeros((17500,feature_length))
test_neg = np.zeros((7500,feature_length))
for i in range(17500):
    train_neg[i] = matrix_doc_neg[k1[i]]
for i in range(7500):
    test_neg[i] = matrix_doc_neg[17500 + k1[i + 17500]]

train_pos = np.zeros((17500,feature_length))
test_pos = np.zeros((7500,feature_length))
for i in range(17500):
    train_pos[i] = matrix_doc_pos[k1[i]]
for i in range(7500):
    test_pos[i] = matrix_doc_pos[17500 + k1[i + 17500]]
# training set, 标记分两类，第一类为positive， 第二类为negative, 数量相等
train_set = np.zeros(17500 * 2, feature_length)
train_set[0:17500] = train_neg
train_set[17500:] = train_pos
k3 = np.random.permutation(17500 * 2)
train_x = np.zeros(17500 * 2, feature_length)
for i in range(17500*2):
    train_x[i] = train_set[k3[i]]
train_y = np.zeros(17500*2, 2)
for i in range(17500*2):
    if k3[i] >= 17500:
        train_y[i][0] = 1
    else:
        train_y[i][1] = 1
        
# test set，标记分两类，第一类为positive， 第二类为negative, 数量相等
test_set = np.zeros(7500 * 2, feature_length)
test_set[0:7500] = test_neg
test_set[7500:] = test_pos
k4 = np.random.permutation(7500 * 2)
test_x = np.zeros(7500 * 2, feature_length)
for i in range(7500*2):
    test_x[i] = test_set[k4[i]]
test_y = np.zeros(7500*2, 2)
for i in range(7500*2):
    if k4[i] >= 7500:
        test_y[i][0] = 1
    else:
        test_y[i][1] = 1


sio.savemat('savelongdata.mat',{'train_x':train_x, 'train_y':train_y, 'test_x':test_x, 'test_y':test_y})








