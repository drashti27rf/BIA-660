#!/usr/bin/env python
# coding: utf-8

# In[6]:


from collections import Counter
import numpy as np


# Que : 1
# Define a function to analyze the frequency of words in a string

def count_token(string):
    
    string = string.lower()
    lis = string.split()
    for i in lis :
        i.strip()
        if len(i) <= 1:
            lis.remove(i)
    count = Counter(lis)
    
    return count

# Que : 2
# Define a class to analyze a document

class Text_Analyzer():
    
    def __init__(self, input_file):
        self.input_file = input_file
        self.token_count = {}
        
    def analyze(self):
        input_file = open(self.input_file, 'r')
        input_file = input_file.read()
        self.token_count = count_token(input_file)
        

# Que : 3
# Define a function to analyze a numpy array

def analyze_tf(arr):
    
    x = np.sum(arr, axis = 1)
    y = x.reshape(-1, 1)
    tf = np.divide(arr, y)
    tf_new = np.where(arr > 0, 1, 0)
    df = np.sum(tf_new, axis = 0)
    tf_idf = np.divide(tf, df)
    top_3 = (-tf_idf).argsort()[:, 0:3]
    
    return tf_idf

# Que : 4
# (Bonus) Segment documents by punctuation

def analyze_docs(arr):
    
    new_token = []
    unique_words = []
    tf_idf = None
    
    for i in arr:
        token = ("".join((char if char.isalnum() else " ") for char in i).split())

        j = 0
        while  j < len(token):
            if len(token[j]) <= 1:
                token.remove(token[j])
            else:
                j += 1

        new_token.append([x.lower() for x in token])

    unique_words = list(set([i for j in new_token for i in j]))
    tf = []

    for i in range(len(new_token)):
        count = []
        for j in unique_words:
            count.append(new_token[i].count(j))
        tf.append(count)

    # print(type(tf))
    tf_idf = analyze_tf(np.array(tf))
    
    return tf_idf, unique_words



if __name__ == "__main__":  
    
# Test Question 1
    text='''COVID-19 vs. the flu: What’s the difference? 
            The flu has been a common comparison to COVID-19, 
            and they both have many of the same symptoms, 
            like fever, cough and muscle aches.'''   
    
    print("\n")
    print("=== Test Q1 ===")
    print(count_token(text))
    
# Test Question 2
    print("\n")
    print("=== Test Q2 ===")
    
    # first get the foo.txt
    docs=["COVID-19 vs. the flu: What’s the difference? The flu has been a common comparison to COVID-19, and they both have many of the same symptoms, like fever, cough and muscle aches.",
          "As we head into fall, this is normally when we’d start to talk about the flu season, but with the added concern of COVID-19, how should you prepare for that?"
          "The flu has been a common comparison to COVID-19. They both have many of the same symptoms, like fever, cough and muscle aches, says Dr. Mike Sevilla of the Family Practice Center of Salem."
          "Those symptoms sound very similar to COVID-19, so how are you going to know the differences if you catch one?"]
    with open('foo.txt', 'w') as f:
        f.writelines(docs)
        
    analyzer=Text_Analyzer("foo.txt")
    analyzer.analyze()
    print(analyzer.token_count)
    
# Test Question 3
    print("\n")
    print("=== Test Q3 ===")
    
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}) # for pretty print
    
    arr=np.array([[0, 3, 1, 2, 0], [1, 0, 0, 1, 1], [2, 0, 0, 2, 1], [1, 0, 1, 1, 2]])
    print(arr)
    tf_idf=analyze_tf(arr)
    print(tf_idf)
    
# Test Question 4
    print("\n")
    print("=== Test Q4 ===")
    tf_idf, words = analyze_docs(docs)
    print(words)
    print(tf_idf)

