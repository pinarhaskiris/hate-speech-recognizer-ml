import pandas as pd
import numpy as np
import nltk
import random
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, svm, naive_bayes
from sklearn.metrics import accuracy_score
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
df=pd.read_csv('data.csv',sep=";", encoding='cp1252')
df = df.sample(frac=1).reset_index(drop=True)
df.head()

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
for index,entry in enumerate(df['sentence']):
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
    df.loc[index,'text_final'] = str(Final_words)

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['text_final'],df['class'],test_size=0.3)

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)


#Tfidf Vectorizer
Tfidf_vect = TfidfVectorizer()
Tfidf_vect.fit(df['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

'''
#Count Vectorizer
Count_vect = CountVectorizer()
Count_vect.fit(df['text_final'])
Train_X_Count = Count_vect.transform(Train_X)
Test_X_Count = Count_vect.transform(Test_X)
'''

#With Tdidf Vectorizer
# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()

Naive.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score with Tdidf Vectorizer -> ",accuracy_score(predictions_NB, Test_Y)*100)

'''
#With Count Vectorizer
# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Count,Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Count)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score with Count Vectorizer -> ",accuracy_score(predictions_NB, Test_Y)*100)
'''