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
from sklearn import model_selection, svm, naive_bayes, metrics
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import nltk
import pickle 
import csv

# read the processed data
dp = pd.read_csv('processed_data_vol2.csv', encoding='cp1252')

# read the processed data
dc = pd.read_csv('class.csv', encoding='cp1252')

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(dp['text_final'], dc['class'], test_size=0.3)

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

#With Tfidf Vectorizer
Tfidf_vect = TfidfVectorizer()
Tfidf_vect.fit(dp['text_final'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

# ---------------- SVM ----------------
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)

# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score with Tdidf Vectorizer -> ",accuracy_score(predictions_SVM, Test_Y)*100)

# save the trained SVM model to disk
filename = 'finalized_model_SVM.sav'
pickle.dump(SVM, open(filename, 'wb'))

# ---------------- NAIVE BAYES ---------

# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score with Tdidf Vectorizer -> ", accuracy_score(predictions_NB, Test_Y)*100)

# save the trained naive bayes model to disk
filename = 'finalized_model_NB.sav'
pickle.dump(Naive, open(filename, 'wb'))

# PLOT CONFUSION MATRIXES
def generate_conf_matrixes(model, predictions, analyzer, vectorizer):
    mat = confusion_matrix(predictions, Test_Y)
    axis_labels=['Hateful', 'Not Hateful']
    sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=axis_labels, yticklabels=axis_labels)
    plt.title(f"{model} with {vectorizer} ({analyzer} based)")
    plt.xlabel('Predicted Categories')
    plt.ylabel('True Categories')
    plt.show() 


# SVM
generate_conf_matrixes("SVM", predictions_SVM, "word", "TFIDF")

# Naive Bayes
generate_conf_matrixes("Naive Bayes", predictions_NB, "word", "TFIDF")
