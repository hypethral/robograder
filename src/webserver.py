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
from nltk.tokenize import word_tokenize, sent_tokenize; # nltk.download('punkt'); nltk.download('averaged_perceptron_tagger');
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
    print(essay)
    number_of_sentences = sent_tokenize(essay)
    return len(number_of_sentences)
def pos(essay):
    
    return pos
def count_books(essay):
    data = open('myBook.txt').read()
    book = re.findall('[a-z]+', data.lower())
    essay=essay.lower()
    new_essay = re.sub("[^A-Za-z0-9]"," ",essay)
    new_essay = re.sub("[0-9]","",new_essay)
    count=0
    all_words = new_essay.split()
    for i in all_words:
        if i in book:
            count+=1
    return count
import spacy # python3 -m spacy download en_core_web_lg
#load the large English model
nlp = spacy.load("en_core_web_lg")

def entity_recognition_ORG(essay):
    doc = nlp(essay)
    c=0;
    for ent in doc.ents:
        if ent.label_ == "ORG":
            c += 1;
    return c
def entity_recognition_PERSON(essay):
    doc = nlp(essay)
    c=0;
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            c += 1;
    return c
def entity_recognition_NORP(essay):
    doc = nlp(essay)
    c=0;
    for ent in doc.ents:
        if ent.label_ == "NORP":
            c += 1;
    return c
def entity_recognition_FAC(essay):
    doc = nlp(essay)
    c=0;
    for ent in doc.ents:
        if ent.label_ == "FAC":
            c += 1;
    return c
def entity_recognition_GPE(essay):
    doc = nlp(essay)
    c=0;
    for ent in doc.ents:
        if ent.label_ == "GPE":
            c += 1;
    return c

def entity_recognition_LOC(essay):
    doc = nlp(essay)
    c=0;
    for ent in doc.ents:
        if ent.label_ == "LOC":
            c += 1;
    return c
def entity_recognition_PRODUCT(essay):
    doc = nlp(essay)
    c=0;
    for ent in doc.ents:
        if ent.label_ == "PRODUCT":
            c += 1;
    return c
def entity_recognition_EVENT(essay):
    doc = nlp(essay)
    c=0;
    for ent in doc.ents:
        if ent.label_ == "EVENT":
            c += 1;
    return c
def entity_recognition_WORK_OF_ART(essay):
    doc = nlp(essay)
    c=0;
    for ent in doc.ents:
        if ent.label_ == "WORK_OF_ART":
            c += 1;
    return c
def entity_recognition_DATE(essay):
    doc = nlp(essay)
    c=0;
    for ent in doc.ents:
        if ent.label_ == "DATE":
            c += 1;
    return c
def entity_recognition_QUANTITY(essay):
    doc = nlp(essay)
    c=0;
    for ent in doc.ents:
        if ent.label_ == "QUANTITY":
            c += 1;
    return c
def entity_recognition_CARDINAL(essay):
    doc = nlp(essay)
    c=0;
    for ent in doc.ents:
        if ent.label_ == "CARDINAL":
            c += 1;
    return c
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
    essay = re.sub("[^A-Za-z.!? ]","",essay)
    return essay
stop_words = set(stopwords.words('english')) 
def remove_stop_words(essay):
    word_tokens = word_tokenize(essay) 
    filtered_sentence = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w)
    return ' '.join(filtered_sentence)

def use_aes_lower(essay):
    if len(essay) > 30:
        rf = pickle.load(open("Models/RFBagLower",'rb'))
        vectorizer = pickle.load(open("Models/vectorizerLower",'rb'))
        essay = remove_puncs(essay)
        clean_essay = remove_stop_words(essay)
        myDataset = pd.DataFrame([essay], columns=['essay'])
        myDataset['clean_essay'] = clean_essay
        # Convert to lower str
        myDataset['clean_essay'] = myDataset['clean_essay'].str.lower()
        import spacy # python3 -m spacy download en_core_web_lg
        #load the large English model
        nlp = spacy.load("en_core_web_lg")
        #list to store the tokens and pos tags 
        token = []
        pos = []
        for sent in nlp.pipe(myDataset['essay']):
            if sent.has_annotation('DEP'):
                #add the tokens present in the sentence to the token list
                token.append([word.text for word in sent])
                #add the pos tage for each token to the pos list
                pos.append([word.pos_ for word in sent])
        myFeatures = myDataset.copy()
        myFeatures['pos'] = pos
        myFeatures['adjective'] = myFeatures.apply(lambda x: x['pos'].count('ADJ'), axis=1)
        myFeatures['adposition'] = myFeatures.apply(lambda x: x['pos'].count('ADP'), axis=1)
        myFeatures['adverb'] = myFeatures.apply(lambda x: x['pos'].count('ADV'), axis=1)
        myFeatures['auxiliary'] = myFeatures.apply(lambda x: x['pos'].count('AUX'), axis=1)
        myFeatures['conjunction'] = myFeatures.apply(lambda x: x['pos'].count('CONJ'), axis=1)
        myFeatures['determiner'] = myFeatures.apply(lambda x: x['pos'].count('DET'), axis=1)
        myFeatures['interjection'] = myFeatures.apply(lambda x: x['pos'].count('INTJ'), axis=1)
        myFeatures['noun'] = myFeatures.apply(lambda x: x['pos'].count('NOUN'), axis=1)
        myFeatures['pronoun'] = myFeatures.apply(lambda x: x['pos'].count('PRON'), axis=1)
        myFeatures['proper-noun'] = myFeatures.apply(lambda x: x['pos'].count('PROPN'), axis=1)
        myFeatures['punctuation'] = myFeatures.apply(lambda x: x['pos'].count('PUNCT'), axis=1)
        myFeatures['verb'] = myFeatures.apply(lambda x: x['pos'].count('VERB'), axis=1)
        myFeatures['entity_recognition_ORG'] = myFeatures['essay'].apply(entity_recognition_ORG)
        myFeatures['entity_recognition_PERSON'] = myFeatures['essay'].apply(entity_recognition_PERSON)
        myFeatures['entity_recognition_NORP'] = myFeatures['essay'].apply(entity_recognition_NORP)
        myFeatures['entity_recognition_FAC'] = myFeatures['essay'].apply(entity_recognition_FAC)
        myFeatures['entity_recognition_GPE'] = myFeatures['essay'].apply(entity_recognition_GPE)
        myFeatures['entity_recognition_LOC'] = myFeatures['essay'].apply(entity_recognition_LOC)
        myFeatures['entity_recognition_PRODUCT'] = myFeatures['essay'].apply(entity_recognition_PRODUCT)
        myFeatures['entity_recognition_EVENT'] = myFeatures['essay'].apply(entity_recognition_EVENT)
        myFeatures['entity_recognition_WORK_OF_ART'] = myFeatures['essay'].apply(entity_recognition_WORK_OF_ART)
        myFeatures['entity_recognition_DATE'] = myFeatures['essay'].apply(entity_recognition_DATE)
        myFeatures['entity_recognition_QUANTITY'] = myFeatures['essay'].apply(entity_recognition_QUANTITY)
        myFeatures['entity_recognition_CARDINAL'] = myFeatures['essay'].apply(entity_recognition_CARDINAL)
        myFeatures['ebooks'] = myFeatures['clean_essay'].apply(count_books)
        myFeatures['Sat500'] = myFeatures['essay'].apply(count_my500)
        myFeatures['char_count'] = myFeatures['essay'].apply(no_of_char)
        myFeatures['word_count'] = myFeatures['essay'].apply(no_of_words)
        myFeatures['sentences_count'] = myFeatures['essay'].apply(sentences_count)
        myFeatures['spelling_mistake_count'] = myFeatures['essay'].apply(spell_check_count)
        myFeatures['avg_word_len'] = myFeatures['essay'].apply(avg_word_len)
        cv = vectorizer.transform(myDataset['clean_essay'])
        X = cv.toarray()
        X = np.concatenate((myFeatures.iloc[:, 3:], X), axis = 1)
        myCheck = pd.DataFrame(X)
        y_pred = rf.predict(X)
        predict = (y_pred[0])
        print(list(myFeatures.iloc[:, 3:].columns))
        print(myCheck)
        print("Score: ", predict)
        return predict
    else:
        return "word_error"



app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template ('lower.html')


score = -1;
@app.route('/', methods=['POST'])
def do_aes_lower():
    myEssay = request.get_json("text")["text"]
    score = use_aes_lower(myEssay)
    return jsonify({'score': score}), 201


@app.route('/upper', methods=['GET'])
def upper():
    return render_template ('upper.html')


score = -1;
@app.route('/upper', methods=['POST'])
def do_aes_upper():
    myEssay = request.get_json("text")["text"]
    score = use_aes_lower(myEssay)
    return jsonify({'score': score}), 201
app.run(host="127.0.0.1", port=8080,debug=True)



