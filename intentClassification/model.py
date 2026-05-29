# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 17:02:19 2021

@author: Niraj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib as jb
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

# defining unwanted intents
unwanted_intent = ['vague-price', 'insist', 'inform']

train_df.drop(train_df[(train_df['intent']=='insist') | (train_df['intent']=='vague-price')
                     | (train_df['intent']=='inform')].index, inplace = True)
test_df.drop(test_df[(test_df['intent']=='insist') | (test_df['intent']=='vague-price')
                     | (test_df['intent']=='inform')].index, inplace = True)

unknown_intent_list= train_df.index[train_df['intent'] == 'counter-price'].tolist()
unknown_intent_list = unknown_intent_list[5000:]
train_df.drop(unknown_intent_list, inplace=True)

cp_intent_list = train_df.index[train_df['intent'] == 'counter-price'].tolist()
cp_intent_list = cp_intent_list[5000:]
train_df.drop(cp_intent_list, inplace=True)

MODEL_FILENAME = 'ic_model.pkl'
VECT_FILENAME = 'vectorizer.pkl'

def count_vector(data):
    count_vectorizer = CountVectorizer()
    count_vectorizer.fit(data.values.astype('U'))
    vect = count_vectorizer.transform(data.values.astype('U'))
    jb.dump(count_vectorizer, VECT_FILENAME)
    return vect, count_vectorizer

def tfidf_vector(data):
    tfidf_vectorizer = TfidfVectorizer()
    vect = tfidf_vectorizer.fit_transform(data.values.astype('U'))
    return vect, tfidf_vectorizer

X_train_count, count_vectorizer = count_vector(train_df["stemmed_text"])
X_train_tfidf, tfidf_vectorizer = tfidf_vector(train_df["stemmed_text"])

X_test_count = count_vectorizer.transform(test_df["stemmed_text"])
X_test_tfidf = tfidf_vectorizer.transform(test_df["stemmed_text"])

print(X_train_count.shape, X_test_count.shape)
print(X_train_tfidf.shape, X_test_tfidf.shape)

random_state = 42

models=[
        XGBClassifier(max_depth=6, n_estimators=500),
        SVC(random_state=random_state, kernel='linear'),
        LogisticRegression(solver = 'sag', random_state=random_state),
        RandomForestClassifier(n_estimators=500,random_state=random_state),
        MultinomialNB(),
        DecisionTreeClassifier(random_state = random_state),
        KNeighborsClassifier(),
       ]

metric = []
# CV = 5
# cv_df = pd.DataFrame(index=range(CV * len(models)))
# entries = []
def fit_and_predict(model,x_train,x_test,y_train,y_test,vectorizer):
    classifier = model
    classifier_name = str(classifier.__class__.__name__)
    eval_set = [(x_test, y_test)]
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    if(classifier_name=='XGBClassifier' and str(vectorizer)=='Count vector'):
        jb.dump(classifier, MODEL_FILENAME)
    # accuracies  = cross_val_score(model, x, y, scoring='accuracy', cv=CV)
    # for fold_idx, accuracy in enumerate(accuracies):
    #         entries.append((str(classifier.__class__.__name__),str(vectorizer), fold_idx, accuracy))

    f1score = f1_score(y_test,y_pred,average='weighted')
    train_accuracy = round(classifier.score(x_train,y_train)*100)
    test_accuracy =  round(accuracy_score(y_test,y_pred)*100)
    
    
    metric.append({
        "model": classifier_name,
        "f1 score": f1score, 
        "train accuracy": train_accuracy, 
        "test accuracy": test_accuracy, 
        "vectorizer": str(vectorizer),
        })

    print(str(classifier.__class__.__name__) +" using "+ str(vectorizer))
    print(classification_report(y_test,y_pred))    
    # print('Accuracy over splitted train and test set')
    print('Accuracy of classifier on training set:{}%'.format(train_accuracy))
    print('Accuracy of classifier on test set:{}%' .format(test_accuracy))
    

for model in models:
    y_train = train_df.encoded_intent
    y_test = test_df.encoded_intent
    # x = X_train_count
    # x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
    fit_and_predict(model,X_train_count,X_test_count,y_train,y_test,'Count vector')
    
    x = X_train_tfidf
    # x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
    fit_and_predict(model,X_train_tfidf,X_test_tfidf,y_train,y_test, 'Tfidf vector')
    

metric_df = pd.DataFrame(metric)
metric_df = metric_df.sort_values('f1 score', ascending=False)
metric_df.to_csv('model_metrics.csv', index=False)
