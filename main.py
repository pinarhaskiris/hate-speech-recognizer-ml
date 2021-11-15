import csv
import re

hateful = []
not_hateful = []

# open file for reading
with open('gab.csv') as dataset:

     # read file as csv file
    csvReader = csv.reader(dataset)

    #collect conversations
    conversations = []
    for row in csvReader:
        match = re.search('1\\. [0-9]+', str(row))
        if (match):
            conversations.append(row)

#collect posts and indexes of hateful ones for each conversation
conversation_parts = []
for i in range(len(conversations)):
    part = (conversations[i][1], conversations[i][2])
    conversation_parts.append(part)

str1 = str(conversations[0][1].split('\n')).replace(r'\t', '')
print(str1)




    