# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 20:55:30 2021

@author: Niraj
"""
import re
import string
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from datasets import load_dataset
import numpy as np
from numpy.core.numeric import array_equal
import pandas as pd

stop_words = stopwords.words('english')
dataset = load_dataset('craigslist_bargains', split= 'train')
test_dataset = load_dataset('craigslist_bargains', split= 'validation')

# -----------------------------------------------------------------------
# create_df :
# Creates a pandas DataFrame from json data from craigslist dataset
# -----------------------------------------------------------------------
def create_df(dataset):
    text_list = []
    text_price_list = []
    agent_turn_list = []
    intent_list = []
    for i in range(len(dataset)):        
        for j in range(len(dataset[i]['utterance'])):
            if dataset[i]['utterance'][j] == "" or dataset[i]['dialogue_acts']['intent'][j] == "":
                break
            else:
                text_list.append(dataset[i]['utterance'][j])
                agent_turn_list.append(dataset[i]['agent_turn'][j])
                intent_list.append(dataset[i]['dialogue_acts']['intent'][j])
                text_price_list.append(dataset[i]['dialogue_acts']['price'][j])
    text_array = np.array(text_list)
    text_price_array = np.array(text_price_list)
    agent_turn_array = np.array(agent_turn_list)
    intent_array = np.array(intent_list)
    
    data = {'text': text_array, 'text_price': text_price_array,
            'agent_turn': agent_turn_array, 'intent': intent_array}
    df = pd.DataFrame(data)
    return df

test_df = create_df(test_dataset)
df = create_df(dataset)

# Remove all punctuations
def remove_all_punct(text):
    table = str.maketrans('','',string.punctuation)
    return text.translate(table)

# Remove numbers, replace it by NUMBER
def remove_number(text):
    num = re.compile(r'[-+]?[.\d]*[\d]+[:,.\d]*')
    return num.sub(r'NUMBER', text)

# -----------------------------------------------------------------------
# text_preprocess :
# Makes text lower, removes all punctuation, removes number and replaces
# it with string "NUMBER", tokenizes the text and then removes stop words.
# -----------------------------------------------------------------------
def text_preprocess(text):
    # porter = PorterStemmer()
    text = remove_all_punct(text)
    text = remove_number(text)
    text = text.lower()
    # text = porter.stem(text)
    return text

def text_stemmer(text):
    stemmer = PorterStemmer()
    text = ' '.join(stemmer.stem(token) for token in word_tokenize(text))
    return text

def text_tokenize(text):
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    return text

def text_tokenize_with_stopwords(text):
    text = word_tokenize(text)
    text = [word for word in text]
    return text

df['preprocessed_text'] = df["text"].apply(lambda x : text_preprocess(x))
test_df['preprocessed_text'] = test_df["text"].apply(lambda x : text_preprocess(x))
                                                    
df['stemmed_text'] = df["preprocessed_text"].apply(lambda x : text_stemmer(x))
test_df['stemmed_text'] = test_df["preprocessed_text"].apply(lambda x : text_stemmer(x))

df['tokens'] = df['preprocessed_text'].apply(lambda x : text_tokenize(x))
test_df['tokens'] = test_df['preprocessed_text'].apply(lambda x : text_tokenize(x))

df['tokens_with_sw'] = df['preprocessed_text'].apply(lambda x : text_tokenize_with_stopwords(x))
test_df['tokens_with_sw'] = test_df['preprocessed_text'].apply(lambda x : text_tokenize_with_stopwords(x))


train_intent_freq = pd.value_counts(df.intent).to_dict()

# -----------------------------------------------------------------------
# Frequency encoding
# It is a way to utilize the frequency of the categories as labels. 
# In the cases where the frequency is related somewhat with the target 
# variable, it helps the model to understand and assign the weight in 
# direct and inverse proportion, depending on the nature of the data.
# -----------------------------------------------------------------------
def frequency_encoding(column, intent_freq):
    for i in range(len(column)):
        column[i] = intent_freq[column[i]]
    return column
  
train_encoded_intent_col = frequency_encoding(df.intent.tolist(),
                                              train_intent_freq)
test_encoded_intent_col = frequency_encoding(test_df.intent.tolist(),
                                              train_intent_freq)

df['encoded_intent'] = train_encoded_intent_col
test_df['encoded_intent'] = test_encoded_intent_col

# Creates a csv of dataframe
df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

    
