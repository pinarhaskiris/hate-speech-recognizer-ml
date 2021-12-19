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

# load the model SVM from disk
loaded_model_svm = pickle.load(open('finalized_model_SVM.sav', 'rb'))

# load the model NB from disk
loaded_model_nb = pickle.load(open('finalized_model_NB.sav', 'rb'))

# read the processed data
dp = pd.read_csv('processed_data_vol2.csv', encoding='cp1252')

# With Tfidf Vectorizer
Tfidf_vect = TfidfVectorizer()
Tfidf_vect.fit(dp['text_final'])


# GUI

root= tk.Tk()

canvas = tk.Canvas(root, width = 1200, height = 720,  relief = 'raised')
canvas.pack()

label = tk.Label(root, text='Hate-Speech Recognizer')
label.config(font=('helvetica', 28, 'bold'))
canvas.create_window(600, 155, window=label)

label = tk.Label(root, text='Enter sentence:')
label.config(font=('helvetica', 24, 'bold'))
canvas.create_window(600, 250, window=label)

entry = tk.Entry(root, font=('helvetica', 18)) 
canvas.create_window(600, 300, window=entry)

def formatPrediction(model, output, index, user_input):
    label = tk.Label(root, text=f"'{user_input}' is recognized as: ", font=('helvetica', 24), width=70, height=3)
    canvas.create_window(600, 440, window=label)

    labelPred = tk.Label(root, text="", width=20, height=3, font=('helvetica', 18))

    if (output == 0):
        labelPred.config(text=f"{model}: Hateful") 
        labelPred.config(bg="red")

    else:
        labelPred.config(text=f"{model}: Not Hateful") 
        labelPred.config(bg="green")

    canvas.create_window(600, (480 + (55 * index)), window=labelPred)
    

def predictInput():

    user_input = entry.get() # get input sentence

    new_input = [user_input] # put input sentence in array format (to use in prediction)
    new_input_Tfidf = Tfidf_vect.transform(new_input) # vectorize input

    # SVM prediction
    new_output_svm = loaded_model_svm.predict(new_input_Tfidf)
    # Naive Bayes prediction
    new_output_nb = loaded_model_nb.predict(new_input_Tfidf)
    
    # configure the prediction labels
    formatPrediction("SVM", new_output_svm, 1, user_input)
    formatPrediction("Naive Bayes", new_output_nb, 2, user_input)
    
button = tk.Button(text='Get Predictions', command=predictInput, bg='white', fg='black', font=('helvetica', 19, 'bold'))
canvas.create_window(600, 350, window=button)

root.mainloop()


