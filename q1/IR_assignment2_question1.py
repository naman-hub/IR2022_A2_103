#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[243]:


import os
import numpy as np
import pandas as pd
import nltk
import copy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# ## Preprocessing Steps

# In[162]:


def importDocument(path):
    content = {}
    file_list = []
    for info in os.walk(path):
        filenames = info[2]
        for file in filenames:
            try:
                with open(path+file) as f:
                    lines = f.readlines()
                    content[file] = lines
                    file_list.append(file)
            except:
                print("Discarded file : \t",file)
    return content, file_list


# In[163]:


def onlyWords(documents):
    for key, value in documents.items():
        buff = []
        for line in range(len(value)):
            if len(value[line].strip()) != 0:
                linetoken = nltk.RegexpTokenizer(r"\w+").tokenize(value[line])
                linetoken = [i.lower() for i in linetoken]
                if len(linetoken) != 0:
                    buff.append(linetoken)
        documents[key] = buff
    return documents


# In[164]:


def removeStopWords(documents):
    stop_words = set(stopwords.words('english'))
    for key, value in documents.items():
        for line in range(len(value)):
            value[line] = [i for i in value[line] if not i in stop_words]
        documents[key] = value
    return documents


# In[165]:


def lemmatization(documents):
    lemmatizer = WordNetLemmatizer()
    for key, value in documents.items():
        for line in range(len(value)):
            value[line] = [lemmatizer.lemmatize(i) for i in value[line]]
    return documents


# In[186]:


def removeUnderscore(documents):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    for name, document in documents.items():
        for idx, line in enumerate(document):
            add = []
            for i, word in enumerate(line):
                if not word.isalnum():
                    change = word.replace("_"," ").strip()
                    change = nltk.RegexpTokenizer(r"\w+").tokenize(change)
                    change = [i.lower() for i in change]
                    change = [i for i in change if not i in stop_words]
                    change = [lemmatizer.lemmatize(i) for i in change]
                    add += change
                    line[i] = ""
            line += add
            document[idx] = [word for word in line if len(word) > 0]
        documents[name] = [line for line in document if len(line) > 0]
        
    for key, content in documents.items():
        buffer = []
        for line in content:
            buffer += line
        documents[key] = buffer
        
    return documents


# In[201]:


def queryPreprocess(query):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    linetoken = nltk.RegexpTokenizer(r"\w+").tokenize(query)
    linetoken = [i.lower() for i in linetoken]
    linetoken = [i for i in linetoken if not i in stop_words]
    linetoken = [lemmatizer.lemmatize(i) for i in linetoken]
    return linetoken


# In[228]:


path = "dataset/Humor,Hist,Media,Food/"
documents, files = importDocument(path)


# In[229]:


documents = onlyWords(documents)
documents = removeStopWords(documents)
documents = lemmatization(documents)
documents = removeUnderscore(documents)


# ## A) Jaccard Coefficient

# In[190]:


def jaccard(doc, query):
    intersect = len(set(query).intersection(doc))
    union = len(doc) + len(query) - intersect
    return intersect/union


# In[198]:


def computeScoreJaccard(query, documents, files):
    score = {}
    for name in files:
        score[name] = jaccard(documents[name], query)
    score = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
    return score[0:5]


# In[204]:


n = int(input())
for i in range(n):
    query = str(input("Enter query : "))
    query = queryPreprocess(query)
    ans = computeScore(query, documents, files)
    print("Top 5 documents")
    for tup in ans:
        print(tup[0],"\t",tup[1])
    print()


# ## B) TF-IDF

# In[230]:


def uniqueWords(documents):
    bow = {}
    word_list = {}
    buffer = []
    for i, t in enumerate(documents.items()):
        filename = t[0]
        value = t[1]
        bow[filename] = {}
        buffer += value
        for word in value:
            if word not in bow[filename]:
                bow[filename][word] = 1
            else:
                bow[filename][word] += 1

    unique = sorted(list(set(buffer)))
    for i, word in enumerate(unique):
        word_list[word] = i
    
    return word_list, bow


# In[236]:


def TF(word_dict, cat):
    tf = {}
    for filename, words in word_dict.items():
        tf[filename] = {}
        m = 0
        tot = 0
        for word, count in words.items():
            m = max(m,count)
            tot += count
            if word not in tf[filename]:
                tf[filename][word] = 0
                
            if cat == "Binary":
                tf[filename][word] = 1
            elif cat == "Raw count" or cat == "Term frequency":
                tf[filename][word] = count
            elif cat == "Log normalization":
                tf[filename][word] = np.log10(1+count)
            elif cat == "Double normalization":
                tf[filename][word] = 0.5*count
                
        for word, count in words.items():       
            if cat == "Term frequency":
                tf[filename][word] = tf[filename][word]/tot
            elif cat == "Double normalization":
                tf[filename][word] = 0.5 + (tf[filename][word]/m)

    return tf


# In[249]:


def DF(word_dict, size):
    df = {}
    idf = {}
    for filename, val in word_dict.items():
        for word, count in val.items():
            if word not in df:
                df[word] = [filename]
            else:
                if filename not in df[word]:
                    df[word].append(filename)
                    
    for word, filenames in df.items():
        df[word] = len(filenames)
        idf[word] = np.log10(size/(df[word]+1))
    
    return df, idf


# In[252]:


def TFIDF(tf,idf):
    tf_idf = copy.deepcopy(tf)
    for filename, content in tf_idf.items():
        for word, value in content.items():
            tf_idf[filename][word] = tf[filename][word]*idf[word]
    return tf_idf


# In[258]:


def computeScoreTFIDF(query, files, cat_tf_idf):
    feature = {}
    for cat, val in cat_tf_idf.items():
        feature_vec = {}
        for token in query:
            for file in files:
                if token in val[file]:
                    if file in feature_vec:
                        feature_vec[file] += val[file][token]
                    else:
                        feature_vec[file] = val[file][token]
        feature_vec = sorted(feature_vec.items(), key=lambda kv: kv[1], reverse=True)
        feature[cat] = feature_vec[0:5]
    return feature


# In[250]:


word_list, word_dict = uniqueWords(documents)
df, idf = DF(word_dict, len(documents))


# In[254]:


category = ["Binary", "Raw count", "Term frequency", "Log normalization","Double normalization"]
cat_tfidf = {}
for cat in category:
    tf = TF(word_dict, cat)
    tf_idf = TFIDF(tf,idf)
    cat_tfidf[cat] = tf_idf


# In[265]:


n = int(input())
for i in range(n):
    query = str(input("Enter query : "))
    query = queryPreprocess(query)
    ans = computeScoreTFIDF(query, files, cat_tfidf)
    print("Top 5 documents")
    for cat, val in ans.items():
        print(cat + ":")
        for tup in val:
            print(tup[0],"\t",tup[1])
        print()

