# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 00:20:41 2016

@author: ASUS
"""
f1 = open('token_data_pos.txt','r')
f2 = open('token_data_neg.txt','r')
f = open('token_data.txt','r')
## feature extraction
# word tokenizing, then we get dictionary of key = word and value = number of each word
dic_pos = {}
documents_pos = ' '.join(f1.readlines()).split()
for word in documents_pos:
    if word in dic_pos:
        dic_pos[word] += 1
    else:
        dic_pos[word] = 1
        
dic_neg = {}
documents_neg = ' '.join(f2.readlines()).split()
for word in documents_neg:
    if word in dic_neg:
        dic_neg[word] += 1
    else:
        dic_neg[word] = 1

dic_documents = {}
documents = ' '.join(f.readlines()).split()
for word in documents:
    if word in dic_documents:
        dic_documents[word] += 1
    else:
        dic_documents[word] = 1
f1.close()
f2.close()
f.close()

# delet punctuation, numbers, letters
tuple_doc = sorted(dic_documents.items(), key = lambda d:d[0], reverse = False)
#print(tuple_doc[554:20255])
dic_documents = dict(tuple_doc[550:20255])



#print(len(dic_documents))
#print(dic_documents)
#==============================================================================
##上一步删除后剩余19705个非重复词，删除前1.5%的高频词汇剩余19409
tuple_doc = sorted(dic_documents.items(), key = lambda d:d[1], reverse = False) 
#print(tuple_doc)
dic_documents = dict(tuple_doc[0:19409])

#sio.savemat('documents.mat',dic_documents)
#sio.savemat('dic_neg.mat',dic_neg)
#sio.savemat('dic_pos.mat',dic_pos)

##chi-square feature selection
dic_feature_score = {}
for word in dic_documents:
    E = dic_documents[word] * 0.5
    if word in dic_pos:
        P = dic_pos[word]
    else:
        P = 0
    if word in dic_neg:
        N = dic_neg[word]
    else:
        N = 0
    
    dic_feature_score[word] = ((P - E)**2)/E
    
tuple_feature_score = sorted(dic_feature_score.items(), key = lambda d:d[1], reverse = True)
#print(tuple_feature_score)
dic_feature_score = dict(tuple_feature_score[:4000])
word_list = sorted(dic_feature_score)
f = open('word_list_unstem4.txt','r+')
for word in word_list:
    f.write(word + ' ')

f.close()

    
