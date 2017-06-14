# -*- coding: utf-8 -*-
f1 = open('token_data_pos.txt','r')
f2 = open('token_data_neg.txt','r')
f = open('token_data.txt','r')
## feature extraction ###################################
# word tokenizing, then we get dictionary of key = word and value = number of each word
# delet letters and lemmatizering
from nltk.stem.wordnet import WordNetLemmatizer
wnl = WordNetLemmatizer()
dic_pos = {}
documents_pos = ' '.join(f1.readlines()).split()
docu_pos = []
for word in documents_pos:
    try:
        s = wnl.lemmatize(word)
        docu_pos.append(s.encode())
    except UnicodeDecodeError:
        continue
for word in docu_pos:
    if len(word)==1:
        continue
    elif word in dic_pos:
        dic_pos[word] += 1
    else:
        dic_pos[word] = 1
        
dic_neg = {}
documents_neg = ' '.join(f2.readlines()).split()
docu_neg = []
for word in documents_neg:
    try:
        s = wnl.lemmatize(word)
        docu_neg.append(s.encode())
    except UnicodeDecodeError:
        continue
for word in docu_neg:
    if len(word)==1:
        continue
    elif word in dic_neg:
        dic_neg[word] += 1
    else:
        dic_neg[word] = 1

dic_documents = {}
documents = ' '.join(f.readlines()).split()
docu = []
for word in documents:
    try:
        s = wnl.lemmatize(word)
        docu.append(s.encode())
    except UnicodeDecodeError:
        continue
for word in docu:
    if len(word)==1:
        continue
    elif word in dic_documents:
        dic_documents[word] += 1
    else:
        dic_documents[word] = 1
f1.close()
f2.close()
f.close()

# delet punctuation, numbers
tuple_doc = sorted(dic_documents.items(), key = lambda d:d[0], reverse = False)
dic_documents = dict(tuple_doc[531:18551])

# now there are 18021 words in dictionary, then delet most 1.5% frequent words
tuple_doc = sorted(dic_documents.items(), key = lambda d:d[1], reverse = False) 
# 18021 * 1.5% = 270
dic_documents = dict(tuple_doc[0:17752])


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
dic_feature_score = dict(tuple_feature_score[:6000])
word_list = sorted(dic_feature_score)
word_list = dic_documents.keys()
f = open('word_list2.txt','r+')
for word in word_list:
    f.write(word + ' ')

f.close()

    
