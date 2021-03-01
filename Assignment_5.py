#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from nltk.cluster import KMeansClusterer, cosine_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import mixture
import numpy as np
from sklearn import metrics

def cluster_docs(docs, num_clusters=2, model_type = 1):
    
    model, tfidf_vect = None, None
    tfidf_vect = TfidfVectorizer(stop_words="english",min_df=5) 
    dtm = tfidf_vect.fit_transform(docs)
    
    if model_type == 1:
        model = KMeansClusterer(num_clusters, cosine_distance, repeats=20)
        cluster = model.cluster(dtm.toarray(), assign_clusters=True)
        
    elif model_type == 2:
        model = KMeans(n_clusters=num_clusters, n_init=20, random_state = 42).fit(dtm)
        
    else:
        model = mixture.GaussianMixture(n_components=num_clusters, covariance_type='diag', random_state=42, n_init=20).fit(dtm.toarray())
        
    return model, tfidf_vect

def interpret_cluster (model, TfidfVect, model_type):
    
    result = []
    
    if model_type == 1:
        centroids=np.array(model.means())
    elif model_type == 2:
        centroids = np.array(model.cluster_centers_)
    else:
        centroids = np.array(model.means_)
    
    sorted_centroids = centroids.argsort()[:, ::-1] 
    voc_lookup= tfidf_vect.get_feature_names()
        
    for i in range(5):
    # get words with top 20 tf-idf weight in the centroid
        result.append([voc_lookup[word_index] for word_index in sorted_centroids[i, :20]])
    return result

def evaluate_model(test_docs, labels, model, model_type, TfidfVect):
    
    test_dtm = TfidfVect.transform(test_docs)
    
    if model_type == 1:
        predicted = [model.classify(v) for v in test_dtm.toarray()]
        
    else:
        predicted = model.predict(test_dtm.toarray())
    
    confusion_df = pd.DataFrame(list(zip(labels.values, predicted)),                            columns = ["label", "cluster"])
    
    confusion_matrix = pd.crosstab( index=confusion_df.cluster, columns=confusion_df.label)
    
    matrix = confusion_matrix.idxmax(axis=1)
    
    predicted_target=[matrix[i]                   for i in predicted]

    print(metrics.classification_report(labels, predicted_target))

    return (pd.DataFrame({'cluster':matrix.index, 'label':matrix.values}))



if __name__ == "__main__":
    
    train = pd.read_csv("hw5_train.csv")
    test = pd.read_csv("hw5_test.csv")
    
    
    model_dict = {1: "KMeans with Cosine Distance",
                 2: "KMeans with Euclidean Distance",
                 3: "GMM"}
    
    for model_type in [1,2,3]:
        print("\n")
        print("======= " + model_dict[model_type]+" =======")
        print("\n")
        
        # Q1
        model, tfidf_vect = cluster_docs(train["text"], num_clusters=5, model_type = model_type)
        
        # Q2
        results = interpret_cluster (model, tfidf_vect, model_type = model_type)        
        for i, words in enumerate(results):
            print(i, ' '.join(words),"\n")
        
        # Q3
        evaluate_model(test["text"], test["label"], model=model, model_type =model_type, TfidfVect = tfidf_vect)

