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

# load the model SVM from disk
loaded_model_svm = pickle.load(open('finalized_model_SVM.sav', 'rb'))

# load the model NB from disk
loaded_model_nb = pickle.load(open('finalized_model_NB.sav', 'rb'))

# read the processed data
dp = pd.read_csv('processed_data_vol2.csv', encoding='cp1252')

# With Tfidf Vectorizer
Tfidf_vect = TfidfVectorizer()
Tfidf_vect.fit(dp['text_final'])

user_input = input("Please enter sentence: ")

new_input = [user_input]
new_input_Tfidf = Tfidf_vect.transform(new_input)

# SVM prediction
new_output_svm = loaded_model_svm.predict(new_input_Tfidf)
print("PREDICTION OF NEW INPUT (SVM):", new_output_svm)

# Naive Bayes prediction
new_output_nb = loaded_model_nb.predict(new_input_Tfidf)
print("PREDICTION OF NEW INPUT (NB):", new_output_nb)


