import pandas as pd
import numpy as np

columns = ['sentence', 'class']

train = []
num_of_hateful = 0
num_of_nothateful = 0
total_num_of_sentences = 0

with open("train.txt") as trainFile:
    for line in trainFile:
      line = line.split('%&QQ@')
      line[1] = line[1].replace("\n", "")
      train.append(line)

      if line[1].strip() == "hatefull":
        num_of_hateful += 1
      else:
        num_of_nothateful += 1

      total_num_of_sentences += 1

prob_hateful = num_of_hateful/total_num_of_sentences
prob_nothateful = num_of_nothateful/total_num_of_sentences

training_data = pd.DataFrame(train, columns=columns)
training_data

from sklearn.feature_extraction.text import CountVectorizer

hatefull_docs = [train['sentence'] for index, train in training_data.iterrows() if train['class'] == 'hatefull']

vec_hatefull = CountVectorizer()
X_hatefull = vec_hatefull.fit_transform(hatefull_docs)
tdm_hatefull = pd.DataFrame(X_hatefull.toarray(), columns=vec_hatefull.get_feature_names())

tdm_hatefull

nothatefull_docs = [train['sentence'] for index, train in training_data.iterrows() if train['class'] == 'nothatefull']

vec_nothatefull = CountVectorizer()
X_nothatefull = vec_nothatefull.fit_transform(nothatefull_docs)
tdm_nothatefull = pd.DataFrame(X_nothatefull.toarray(), columns=vec_nothatefull.get_feature_names())

tdm_nothatefull

word_list_hatefull = vec_hatefull.get_feature_names();    
count_list_hatefull = X_hatefull.toarray().sum(axis=0) 
freq_hatefull = dict(zip(word_list_hatefull,count_list_hatefull))
freq_hatefull

word_list_nothatefull = vec_nothatefull.get_feature_names();    
count_list_nothatefull = X_nothatefull.toarray().sum(axis=0) 
freq_nothatefull = dict(zip(word_list_nothatefull,count_list_nothatefull))
freq_nothatefull

from sklearn.feature_extraction.text import CountVectorizer

docs = [lines['sentence'] for index,lines in training_data.iterrows()]

vec = CountVectorizer()
X = vec.fit_transform(docs)

total_features = len(vec.get_feature_names())
total_features

total_cnts_features_hatefull = count_list_hatefull.sum(axis=0)
total_cnts_features_nothatefull = count_list_nothatefull.sum(axis=0)
print("Total features of hatefull class " + str(total_cnts_features_hatefull))
print("Total features of nothatefull class " + str(total_cnts_features_nothatefull))

import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize

fp = 0
fn = 0
tn = 0
tp = 0

with open("test.txt") as testFile:
    for line in testFile:
      line = line.split('%&QQ@')
      line[1] = line[1].replace("\n", "")
      new_sentence = line[0]
      new_word_list = word_tokenize(new_sentence)

      prob_hatefull_with_ls = []
      for word in new_word_list:
          if word in freq_hatefull.keys():
              count = freq_hatefull[word]
          else:
              count = 0
          prob_hatefull_with_ls.append((count + 1)/(total_cnts_features_hatefull + total_features))
      dict(zip(new_word_list,prob_hatefull_with_ls))

      hatefull_prob = 1
      for prob in prob_hatefull_with_ls:
        hatefull_prob = hatefull_prob * prob

      prob_nothatefull_with_ls = []
      for word in new_word_list:
          if word in freq_nothatefull.keys():
              count = freq_nothatefull[word]
          else:
              count = 0
          prob_nothatefull_with_ls.append((count + 1)/(total_cnts_features_nothatefull + total_features))
      dict(zip(new_word_list,prob_nothatefull_with_ls))

      nothatefull_prob = 1
      for prob in prob_nothatefull_with_ls:
        nothatefull_prob = nothatefull_prob * prob

      hatefull_prob = prob_hateful * hatefull_prob
      nothatefull_prob = prob_nothateful * nothatefull_prob

      print(f"hateful: {hatefull_prob}")
      print(f"nothateful: {nothatefull_prob}")

      predicted = 0
      if (hatefull_prob > nothatefull_prob):
        predicted = 1 #hateful

          
      if (hatefull_prob > nothatefull_prob and line[1] == "hatefull"):
        tp += 1
      elif (hatefull_prob > nothatefull_prob and line[1] != "hatefull"):
        fp += 1
      elif (hatefull_prob < nothatefull_prob and line[1] == "nothatefull"):
        tn += 1
      elif (hatefull_prob < nothatefull_prob and line[1] != "nothatefull"):
        fn += 1

accuracy = ((tp + tn)/(tp + fp + tn + fn)) * 100
print(accuracy)
print(f"TP: {tp} FP: {fp} TN: {tn} FN: {fn}")

