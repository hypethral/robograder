# For Webserver - Flask
from flask import Flask, request, jsonify, render_template

# Data manipulation(cleaning), Plot 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Pre-processing
import nltk
from nltk.corpus import stopwords; # nltk.download('stopwords');
from nltk import tokenize
from nltk import pos_tag
from nltk.tokenize import word_tokenize; # nltk.download('punkt'); nltk.download('averaged_perceptron_tagger');
import re
from spellchecker import SpellChecker
from collections import Counter

# Prep ML - Split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# ML models
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Extract features from the essay
# sentence_2_word removes punctuation from a sentence and tokenizes it into individual word
def sentence_2_word(x):
    x=re.sub("[^A-Za-z0-9]"," ",x)
    words=nltk.word_tokenize(x)
    return words
# essay2word takes an essay as input, strips all of the white space, 
# tokenizes the essay using the NLTK library, creates a list of words, 
# and then returns the list of words. The sent2word() function essentially turns sentences from the essay into words.
# example My name is Arslan to ('My','Name','is','Arslan')
def essay2word(essay):
    essay = essay.strip()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw = tokenizer.tokenize(essay)
    final_words=[]
    for i in raw:
        if(len(i)>0):
            final_words.append(sentence_2_word(i))
    return final_words
# check_spell_error checks for spelling errors in an essay. It does this by removing any punctuation or numbers from the essay and splitting the essay into separate words. 
# It then checks if each word is present in a list of words (lowecased). If the word is not present, the count of the spelling errors is incremented by one. 
# Finally, the number of spelling errors is returned.
def spell_check_count(essay):
    spell = SpellChecker()
    essay=essay.lower()
    count = 0
    new_essay = re.sub("[^A-Za-z0-9]"," ",essay)
    new_essay = re.sub("[0-9]","",new_essay)
    all_words = new_essay.split()
    # find those words that may be misspelled
    count = len(list(spell.unknown(all_words)))
    return count
# returns the number of words in the essay
def no_of_words(essay):
    count=0
    for i in essay2word(essay):
        count=count+len(i)
    return count
# The function will go through each word in the essay, 
# count the number of characters in each word, 
# and then return the total count of all the characters combined.
def no_of_char(essay):
    count=0
    for i in essay2word(essay):
        for j in i:
            count=count+len(j)
    return count
# avg_word_len calculates the average length of words in an essay. 
# It takes in an essay as an argument, then returns the number of characters divided by the number of words in the essay.
def avg_word_len(essay):
    return no_of_char(essay)/no_of_words(essay)
# sentences_count takes in an essay as an argument and returns the number of sentences in the essay by counting the number of words in the argument.
def sentences_count(essay):
    return len(essay2word(essay))
def count_nouns(essay):
    sentences = essay2word(essay)
    noun_count=0
    for i in sentences:
        pos_sentence = nltk.pos_tag(i)
        for j in pos_sentence:
            pos_tag = j[1]
            if(pos_tag[0]=='N'):
                noun_count+=1
    return noun_count
def count_adjectives(essay):
    sentences = essay2word(essay)
    adj_count=0
    for i in sentences:
        pos_sentence = nltk.pos_tag(i)
        for j in pos_sentence:
            pos_tag = j[1]
            if(pos_tag[0]=='J'):
                adj_count+=1
    return adj_count
def count_verbs(essay):
    sentences = essay2word(essay)
    verb_count=0
    for i in sentences:
        pos_sentence = nltk.pos_tag(i)
        for j in pos_sentence:
            pos_tag = j[1]
            if(pos_tag[0]=='V'):
                verb_count+=1
    return verb_count
def count_adverts(essay):
    sentences = essay2word(essay)
    adverb_count=0
    for i in sentences:
        pos_sentence = nltk.pos_tag(i)
        for j in pos_sentence:
            pos_tag = j[1]
            if(pos_tag[0]=='R'):
                adverb_count+=1
    return adverb_count
data = open('my500.txt').read()
my500 = re.findall('[a-z]+', data.lower())
def count_my500(essay):
    essay=essay.lower()
    new_essay = re.sub("[^A-Za-z0-9]"," ",essay)
    new_essay = re.sub("[0-9]","",new_essay)
    count=0
    all_words = new_essay.split()
    for i in all_words:
        if i in my500:
            count+=1
    return count
# By using regular expression to remove characters that are not A-Z, a-z, or a space from an essay
# This used to remove, special characters, or other extraneous characters that may be present in an essay - Code Injection
def remove_puncs(essay):
    essay = re.sub("[^A-Za-z ]","",essay)
    return essay
stop_words = set(stopwords.words('english')) 
def remove_stop_words(essay):
    word_tokens = word_tokenize(essay) 
    filtered_sentence = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w)
    return ' '.join(filtered_sentence)

def use_aes(essay):
    if len(essay) > 40:
        rf = pickle.load(open("Models/RF",'rb'))
        myDataset = pd.DataFrame([essay], columns=['clean_essay'])
        # Convert to lower str
        myDataset['clean_essay'] = myDataset['clean_essay'].str.lower()
        myFeatures = myDataset.copy()
        myFeatures['Sat500'] = myFeatures['clean_essay'].apply(count_my500)
        myFeatures['char_count'] = myFeatures['clean_essay'].apply(no_of_char)
        myFeatures['word_count'] = myFeatures['clean_essay'].apply(no_of_words)
        myFeatures['sentences_count'] = myFeatures['clean_essay'].apply(sentences_count)
        myFeatures['spelling_mistake_count'] = myFeatures['clean_essay'].apply(spell_check_count)
        myFeatures['avg_word_len'] = myFeatures['clean_essay'].apply(avg_word_len)
        myFeatures['count_nouns'] = myFeatures['clean_essay'].apply(count_nouns)
        myFeatures['count_adjectives'] = myFeatures['clean_essay'].apply(count_adjectives)
        myFeatures['count_adverts'] = myFeatures['clean_essay'].apply(count_adverts)
        myFeatures['count_verbs'] = myFeatures['clean_essay'].apply(count_verbs)
        vectorizer = pickle.load(open("Models/vectorizer",'rb'))
        cv = vectorizer.transform(myDataset['clean_essay'])
        X = cv.toarray()
        X = np.concatenate((myFeatures.iloc[:, 1:], X), axis = 1)
        myCheck = pd.DataFrame(X)
        print(myCheck)
        y_pred = rf.predict(X)
        predict = (y_pred[0])
        print(predict)
        return predict
    else:
        return "word_error"



app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template ('home.html')


score = 5.5
@app.route('/', methods=['POST'])
def do_aes():
    myEssay = request.get_json("text")["text"]
    myEssay = remove_puncs(myEssay)
    myEssay = remove_stop_words(myEssay)
    score = use_aes(myEssay)
    return jsonify({'score': score}), 201

app.run(host="127.0.0.1", port=8080,debug=True)



