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
import csv

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# read data file
df = pd.read_csv('data.csv',sep=";", encoding='cp1252')
df = df.sample(frac=1).reset_index(drop=True)
df.head()

# ---------------- START OF PREPROCESSING ----------------

# Step - a : Remove blank rows if any.
df['sentence'].dropna(inplace=True)
# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
df['sentence'] = [entry.lower() for entry in df['sentence']]
# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
df['sentence']= [word_tokenize(entry) for entry in df['sentence']]
# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index, entry in enumerate(df['sentence']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    df.loc[index, 'text_final'] = str(Final_words)

# ---------------- END OF PREPROCESSING ----------------

# store the processed versions of the input data (to be used in training and predicting)
df['text_final'].to_csv('processed_data_vol2.csv', index = False, header = True)

# store the classes
df['class'].to_csv('class.csv', index = False, header = True)
