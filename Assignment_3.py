#!/usr/bin/env python
# coding: utf-8

# In[456]:


import spacy
import en_core_web_sm 
nlp = en_core_web_sm.load()
from nltk.probability import FreqDist
import numpy as np
from sklearn import preprocessing
from scipy import spatial
import re
from nltk.stem import WordNetLemmatizer
import pandas as pd

def extract(text):
    
    d = []
    a = [i.lstrip() for i in re.findall('([a-zA-Z ]+):', text)]
    b = re.findall('[+0-9]+.[0-9]%', text)
    c = re.findall('in ([a-zA-Z0-9 ]+)', text)

    for i,j,k in zip(a,b,c):
        d.append((i,j,k))

    print(d)
    return None

def tokenize(doc, lemmatized=True, stopword=True, punctuation = True):
    tokens = []
    text = nlp(doc.lower())
    
    if lemmatized:
        if stopword == False:
            if punctuation == False:
                tokens = [token.lemma_ for token in text if not token.is_stop and not token.is_punct]
            else:
                tokens = [token.lemma_ for token in text if not token.is_stop]
                
        if stopword:
            if punctuation == False:
                tokens = [token.lemma_ for token in text if not token.is_punct]
            else:
                tokens = [token.lemma_ for token in text]
            
    if lemmatized == False:
        if stopword == False:
            if punctuation == False:
                tokens = [token.text for token in text if not token.is_stop and not token.is_punct]
            else:
                tokens = [token.text for token in text if not token.is_stop]
                
        if stopword:
            if punctuation == False:
                tokens = [token.text for token in text if not token.is_punct]
            else:
                tokens = [token.text for token in text]
            
    return tokens


def compute_tfidf(docs, lemmatized=True, stopword=True, punctuation=True):
    
    smoothed_tf_idf, smoothed_idf, words = None, None, None
    a = lemmatized
    b = stopword
    c = punctuation
    
    docs_tokens={idx:FreqDist(tokenize(doc,a,b,c))              for idx,doc in enumerate(docs)}
    
    # step 3. get document-term matrix
    dtm=pd.DataFrame.from_dict(docs_tokens, orient="index" )
    dtm=dtm.fillna(0)
      
    words = np.array(list(dtm.columns)) 
    
    # step 4. get normalized term frequency (tf) matrix        
    tf=dtm.values
    doc_len=tf.sum(axis=1)
    tf=np.divide(tf.T, doc_len).T
    
    # step 5. get idf
    df=np.where(tf>0,1,0)
    idf=np.log(np.divide(len(docs),        np.sum(df, axis=0)))+1

    smoothed_idf=np.log(np.divide(len(docs)+1, np.sum(df, axis=0)+1))+1    
    smoothed_tf_idf=tf*smoothed_idf
    
    return smoothed_tf_idf, smoothed_idf, words

def vectorize_doc(doc, words, idf, lemmatized=True, stopword=True, punctuation = True):
    
    vect = None
       
    a = lemmatized
    b = stopword
    c = punctuation 
    
    docs_tokens = tokenize(doc, a, b, c)

    tf = np.zeros(idf.shape)
    
    for i in words:
        if i in docs_tokens:
            tf[list(words).index(i)] += 1 
    
    norm = np.linalg.norm(tf*idf)
    smoothed_tf_idf = tf*idf/norm
    
    return smoothed_tf_idf

def find_answer(doc_vect, tf_idf, docs):
    
    top_docs  = []
    sim = []
    
    for i in tf_idf:
        sim.append(1 - spatial.distance.cosine(doc_vect, i))
        
    top_docs = [docs[j] for j in (sorted(range(len(sim)), key=lambda i: sim[i], reverse=True)[:3])]

    return top_docs 

if __name__ == "__main__":  
    
    # Test Q1
    
    text='''Consumer Price Index:
            +0.2% in Sep 2020

            Unemployment Rate:
            +7.9% in Sep 2020

            Producer Price Index:
            +0.4% in Sep 2020

            Employment Cost Index:
            +0.5% in 2nd Qtr of 2020

            Productivity:
            +10.1% in 2nd Qtr of 2020

            Import Price Index:
            +0.3% in Sep 2020

            Export Price Index:
            +0.6% in Sep 2020'''
    
    print("\n==================\n")
    print("Test Q1")
    print(extract(text))
    
    data=pd.read_csv("covid_qa.csv")
    # concatenate a pair of question and answer as a single doc
    docs = data.apply(lambda x: x["question"] + " " + x["answer"], axis = 1) 
    
    print("\n==================\n")
    print("Test Q2.1 - Try different parameter values to make sure all options work\n")
    
    # Let's tokenize the first document
    doc = docs[0]
    
    print("===Lemmatize words, keep stop words/punctuations===\n")
    tokens = tokenize(doc, lemmatized=True, stopword=True, punctuation = True)
    print(tokens)
    print("\n")
   
    print("===Lemmatize words, remove stop words/punctuations==\n")
    tokens = tokenize(doc, lemmatized=True, stopword=False, punctuation = False)
    print(tokens)
    print("\n")
    
    print("===Do not lemmatize words, remove stop words, but keep punctuations===\n")
    tokens = tokenize(doc, lemmatized=False, stopword=False, punctuation = True)
    print(tokens)
       
    print("\n==================\n")
    print("Test Q2.2")
    tf_idf, idf, words = compute_tfidf(docs, lemmatized=True, stopword=True, punctuation = False)
    print("TF_IDF Shape: ", tf_idf.shape)
    print("IDF Shape: ", idf.shape)
    
    print("\n==================\n")
    print("Test Q2.3 -- You can try different questions related to Covid-19 here")
    doc = 'Is it safe to travel by plane?'  # What kind of masks should I use?
    vect = vectorize_doc(doc, words, idf, lemmatized=True, stopword=True, punctuation = True)
    # print words with non-zero tf_idf weights
    print([(words[idx], i) for idx, i in enumerate(vect) if i>0])
    
    print("\n==================\n")
    print("Test Q2.4")
    answers = find_answer(vect, tf_idf, docs)
    for a in answers:
        print(a, "\n")

