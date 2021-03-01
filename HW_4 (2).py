#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc,precision_recall_curve
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.model_selection import cross_val_score

#Q1

def nb_model(train_docs, train_y, test_docs, test_y,             stop_words='english',              binary=False, use_idf=True):
    
    clf, tfidf_vect = None, None
    
    #Selecting Parameters for TF_IDF vectorizer
    tfidf_vect = TfidfVectorizer(stop_words=stop_words, binary=binary, use_idf=use_idf) 
    
    #Fitting TF_Idf
    tfidf_vectorizer = tfidf_vect.fit_transform(train_docs)
    
    #Fitting NB model
    clf = MultinomialNB().fit(tfidf_vectorizer, train_y)
    
    #Fitting testing data
    tfidf_vect_test = tfidf_vect.transform(test_docs)
    
    #Predicting Testing Data
    predicted=clf.predict(tfidf_vect_test)
    
    #Classificiation Report
    labels=sorted(train_y.unique())
    print(classification_report(test_y, predicted, labels=labels))
    
    #Getting AUC Value
    predict_p=clf.predict_proba(tfidf_vect_test)
    y_pred = predict_p[:,1]
    fpr, tpr, thresholds = roc_curve(test_y, y_pred, pos_label=1)
    print('AUC: '+ str(np.round(auc(fpr, tpr)*100, 2)) + '%')
    
    #Plotting AUC Curve
    plt.figure();
    plt.plot(fpr, tpr, color='darkorange', lw=2);
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--');
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05]);
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate');
    plt.title('AUC of Naive Bayes Model');
    plt.show();
    
    #plotting precision vs recall curve
    precision, recall, thresholds = precision_recall_curve(test_y,                                 y_pred, pos_label=1)
    print('PRC: '+str(np.round(np.mean(precision)*100, 2))+'%')
    plt.figure();
    plt.plot(recall, precision, color='darkorange', lw=2);
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05]);
    plt.xlabel('Recall');
    plt.ylabel('Precision');
    plt.title('PRC of Naive Bayes Model');
    plt.show();
    
    return clf, tfidf_vect

#Q2

def search_para(docs, y):
    
    text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', MultinomialNB())
                   ])
    
    parameters = {'tfidf__use_idf':[True, False],
                  'tfidf__stop_words':[None,"english"],
                  'tfidf__binary': [True, False],
    }
    
    metric =  "accuracy"

    gs_clf = GridSearchCV(text_clf, param_grid=parameters, scoring=metric, cv=5)
    gs_clf = gs_clf.fit(docs, y)
    
    for param_name in gs_clf.best_params_:
        print(param_name,": ",gs_clf.best_params_[param_name])

    print("best acc: " + str(np.round(gs_clf.best_score_*100, 2))+'%')

#Q3

def k_fold(docs, y):
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', binary=True, use_idf=False)
    tfidf_vect = tfidf_vectorizer.fit_transform(docs)
        
    clf = svm.LinearSVC()
    score = []
    
    for k in range(2, 21):
        cv = KFold(n_splits=k, random_state=1, shuffle=True)
        scores = cross_val_score(clf, tfidf_vect, y, scoring='f1_macro', cv=cv, n_jobs=-1)
        score.append(np.mean(scores))

    plt.plot(range(2, 21), score)
    plt.xlabel('K-Fold')
    plt.ylabel('F1-Macro')
    plt.title('F1-Macro vs K-Fold')
    plt.show();
    return None


def show_analysis(doc, model, tfidf_vect):
    
    tfidf_vectorizer = tfidf_vect.transform([doc])
    predicted = model.predict(tfidf_vectorizer)
    print('Conditional Probability of each class:', model.predict_proba(tfidf_vectorizer))
    print("Lable of given document", predicted)


if __name__ == "__main__":  
    
    train = pd.read_csv("hw4_train.csv")
    test = pd.read_csv("hw4_test.csv")

    print("===== Test Q1 ======")
    
    model, tfidf_vect = nb_model(train["text"], train["label"],          test["text"], test["label"])
    
    print("\n")
    
    print("===== Test Q2 ======")
    
    search_para(train["text"], train["label"])
    
    model, tfidf_vect = nb_model(train["text"], train["label"],         test["text"], test["label"],        binary = True,stop_words = None,         use_idf = False)
    
    print("\n")
    
    print("===== Test Q3 ======")
    
    k_fold(train["text"], train["label"])
    
    print("\n")
    
    print("===== Test Q4 ======")
    doc="Great American classic that you can get for free on kindle. I was supposed to read this back in high school but wasn't not interested at the time. But now with the benefit of age I can really appreciate it."
    show_analysis(doc, model, tfidf_vect)

