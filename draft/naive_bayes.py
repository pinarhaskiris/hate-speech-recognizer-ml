import pandas as pd
import numpy as np
import random
import math

all_data = []
columns = ['sentence', 'class']

num_of_hateful = 0
num_of_nothateful = 0
total_num_of_sentences = 0

# get all data
with open("all_data.txt") as trainFile:
    for line in trainFile:
      line = line.split('%&QQ@')
      line[1] = line[1].replace("\n", "")
      all_data.append(line)

      if line[1].strip() == "hateful":
        num_of_hateful += 1
      else:
        num_of_nothateful += 1

      total_num_of_sentences += 1

# shuffle train data
random.shuffle(all_data)

# spare a portion of training data for testing
train_size = math.floor(len(all_data) * 90 / 100)
test = []
train = []

for n in range(0, train_size):
  train.append(all_data[n])

for m in range(train_size, len(all_data)):
  test.append(all_data[m])

prob_hateful = num_of_hateful/total_num_of_sentences
prob_nothateful = num_of_nothateful/total_num_of_sentences

training_data = pd.DataFrame(train, columns=columns)

# extracting features - hateful
from sklearn.feature_extraction.text import CountVectorizer

hateful_docs = [train['sentence'] for index, train in training_data.iterrows() if train['class'] == 'hateful']

vec_hateful = CountVectorizer()
X_hateful = vec_hateful.fit_transform(hateful_docs)
tdm_hateful = pd.DataFrame(X_hateful.toarray(), columns=vec_hateful.get_feature_names())

# extracting features - not hateful
nothateful_docs = [train['sentence'] for index, train in training_data.iterrows() if train['class'] == 'nothateful']

vec_nothateful = CountVectorizer()
X_nothateful = vec_nothateful.fit_transform(nothateful_docs)
tdm_nothateful = pd.DataFrame(X_nothateful.toarray(), columns=vec_nothateful.get_feature_names())

# calculating frequencies - hateful
word_list_hateful = vec_hateful.get_feature_names();    
count_list_hateful = X_hateful.toarray().sum(axis=0) 
freq_hateful = dict(zip(word_list_hateful,count_list_hateful))

# calculating frequencies - not hateful
word_list_nothateful = vec_nothateful.get_feature_names();    
count_list_nothateful = X_nothateful.toarray().sum(axis=0) 
freq_nothateful = dict(zip(word_list_nothateful,count_list_nothateful))

# extracting features - total
docs = [lines['sentence'] for index, lines in training_data.iterrows()]

vec = CountVectorizer()
X = vec.fit_transform(docs)

total_features = len(vec.get_feature_names())

total_cnts_features_hateful = count_list_hateful.sum(axis=0)
total_cnts_features_nothateful = count_list_nothateful.sum(axis=0)

import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize

fp = 0
fn = 0
tn = 0
tp = 0

test_data = open("test.txt", "a") #for debugging

# go through the test data
for line in test:
  new_sentence = line[0]
  new_word_list = word_tokenize(new_sentence)

# probability calculation for hateful
  prob_hateful_with_ls = []
  for word in new_word_list:
      if word in freq_hateful.keys():
          count = freq_hateful[word]
      else:
          count = 0
      prob_hateful_with_ls.append((count + 1)/(total_cnts_features_hateful + total_features))
  dict(zip(new_word_list,prob_hateful_with_ls))

  hateful_prob = 1
  for prob in prob_hateful_with_ls:
    hateful_prob = hateful_prob * prob

# probability calculation for not hateful
  prob_nothateful_with_ls = []
  for word in new_word_list:
      if word in freq_nothateful.keys():
          count = freq_nothateful[word]
      else:
          count = 0
      prob_nothateful_with_ls.append((count + 1)/(total_cnts_features_nothateful + total_features))
  dict(zip(new_word_list,prob_nothateful_with_ls))

  nothateful_prob = 1
  for prob in prob_nothateful_with_ls:
    nothateful_prob = nothateful_prob * prob

  hateful_prob = prob_hateful * hateful_prob
  nothateful_prob = prob_nothateful * nothateful_prob

# accuracy calculation, hateful = positive, not hateful = negative
  if (hateful_prob > nothateful_prob and line[1] == "hateful"):
    tp += 1
    test_data.write(f'{line[0]}%&QQ@{line[1]}---> hateful (TP) \n')

  elif (hateful_prob > nothateful_prob and line[1] == "nothateful"):
    fp += 1
    test_data.write(f'{line[0]}%&QQ@{line[1]} ---> hateful (FP) \n')

  elif (hateful_prob < nothateful_prob and line[1] == "nothateful"):
    tn += 1
    test_data.write(f'{line[0]}%&QQ@{line[1]} ---> nothateful (TN) \n')

  elif (hateful_prob < nothateful_prob and line[1] == "hateful"):
    fn += 1
    test_data.write(f'{line[0]}%&QQ@{line[1]} ---> nothateful (FN) \n')

test_data.close()
accuracy = ((tp + tn)/(tp + fp + tn + fn)) * 100
print(accuracy)
print(f"TP: {tp} FP: {fp} TN: {tn} FN: {fn}")
print(f"Total number of test posts: {len(test)}")

