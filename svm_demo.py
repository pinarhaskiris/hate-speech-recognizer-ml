import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
import nltk
import pickle 

# load the model from disk
loaded_model = pickle.load(open('finalized_model_SVM.sav', 'rb'))

new_input = input('Enter test sentence: ')

# ----------------------- PREPROCESSING

# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
new_input = new_input.lower()

# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
new_input = word_tokenize(new_input)

# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

# Declaring Empty List to store the words that follow the rules for this step
Final_words = []

# Initializing WordNetLemmatizer()
word_Lemmatized = WordNetLemmatizer()

# pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
for word, tag in pos_tag(new_input):
    # Below condition is to check for Stop words and consider only alphabets
    if word not in stopwords.words('english') and word.isalpha():
        word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
        Final_words.append(word_Final)

# ----------------------- END OF PREPROCESSING

#With Tfidf Vectorizer
Tfidf_vect = TfidfVectorizer()
Tfidf_vect.fit(Final_words)
Test_X_Tfidf = Tfidf_vect.transform(new_input)

"""
print('TEST X TFIDF', Test_X_Tfidf)
print('NEW INPUT', new_input)
"""

new_output = loaded_model.predict(Test_X_Tfidf)
print('PREDICTION: ', new_output)
