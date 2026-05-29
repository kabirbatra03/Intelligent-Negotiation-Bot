import re
import os
import string
import pickle
import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import joblib as jb
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier

MODEL_FILENAME = 'ic_model.pkl'
VECT_FILENAME = 'vectorizer.pkl'

# Remove all punctuations
def remove_all_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

# Remove numbers, replace it by NUMBER
def remove_number(text):
    num = re.compile(r'[-+]?[.\d]*[\d]+[:,.\d]*')
    return num.sub(r'NUMBER', text)

# Stemmer function
def text_stemmer(text):
    stemmer = PorterStemmer()
    text = ' '.join(stemmer.stem(token) for token in word_tokenize(text))
    return text

# text preprocessing
def text_preprocess(text):
    text = remove_all_punct(text)
    text = remove_number(text)
    # text = text.lower()
    text = text_stemmer(text)
    return text

# Converts the string into the vectorized form
def vectorize_text(text):
    count_vectorizer = CountVectorizer()
    vect = count_vectorizer.fit_transform(text)
    return vect

{'unknown': 7680, 'counter-price': 7118, 'inquiry': 4570, 'init-price': 4051, 
 'intro': 3747, 'inform': 2141, 'disagree': 1836, 'agree': 1579, 'vague-price': 366, 'insist': 354}

intent_map = { 4051:'init-price',
               7680:'unknown', 
            #   354:'insist', 
               7118:'counter-price', 
               1579:'agree', 
               3747:'intro', 
               4570:'inquiry', 
               1836:'disagree', 
            #   366:'vague-price', 
            #   2141:'inform'
            }

# intent_map = { 0:'init-price',
#               1:'unknown', 
#               2:'insist', 
#               3:'counter-price', 
#               4:'agree', 
#               5:'intro', 
#               6:'inquiry', 
#               7:'disagree', 
#               8:'vague-price', 
#               9:'inform'
#             }

# Model training and saving
def load_model_and_vectorizer():
    if os.path.exists(MODEL_FILENAME):
        model = jb.load(MODEL_FILENAME)
        vect = jb.load(VECT_FILENAME)
    return model, vect

def predict_intent(text):
    preprocessed_text = text_preprocess(text)
    ic_model, vect = load_model_and_vectorizer()
    vectorized_text = vect.transform([preprocessed_text])
    # predicting the intent
    intent = ic_model.predict(vectorized_text)
    print("current intent: ", intent_map[intent[0]])
    return intent_map[intent[0]]

if __name__ == "__main__":
    input_str = input("Enter input: ")
    predicted_intent = predict_intent(input_str)
