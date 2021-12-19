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
import tkinter as tk

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

root= tk.Tk()

canvas = tk.Canvas(root, width = 1200, height = 700,  relief = 'raised')
canvas.pack()

label = tk.Label(root, text='Hate-Speech Recognizer')
label.config(font=('helvetica', 18, 'bold'))
canvas.create_window(600, 225, window=label)

label = tk.Label(root, text='Enter sentence:')
label.config(font=('helvetica', 14, 'bold'))
canvas.create_window(600, 300, window=label)

entry = tk.Entry(root) 
canvas.create_window(600, 340, window=entry)

def predictInput():
    label = tk.Label(root, text= '',font=('helvetica', 14, 'bold'))
    canvas.create_window(600, 530, window=label)

    user_input = entry.get() # get input sentence

    new_input = [user_input] # put input sentence in array format (to use in prediction)
    new_input_Tfidf = Tfidf_vect.transform(new_input) # vectorize input

    # SVM prediction
    new_output_svm = loaded_model_svm.predict(new_input_Tfidf)
    # Naive Bayes prediction
    new_output_nb = loaded_model_nb.predict(new_input_Tfidf)

    label = tk.Label(root, text=  f"{user_input} is recognized as: ",font=('helvetica', 14))
    canvas.create_window(600, 480, window=label)
    
    label = tk.Label(root, text=f"SVM: {new_output_svm}", font=('helvetica', 14, 'bold'))
    canvas.create_window(600, 530, window=label)

    label = tk.Label(root, text=f"Naive Bayes: {new_output_nb}", font=('helvetica', 14, 'bold'))
    canvas.create_window(600, 560, window=label)
    
button = tk.Button(text='Get Predictions', command=predictInput, bg='brown', fg='white', font=('helvetica', 9, 'bold'))
canvas.create_window(600, 400, window=button)

root.mainloop()


