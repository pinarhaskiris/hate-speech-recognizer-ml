import csv
import re
import string


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

non_hateful = open("non_hateful.txt", "a")
hateful = open("hateful.txt", "a")

#selected a conversation for testing
for i in range(len(conversations)): #each conversation
    for convo in conversations[i][1].split('\n'): #each post in the conversation
        convo = re.sub(r'http\S+', '', convo) #remove links
        base = []
        for j in range(len(convo)): #for each char in a post
            if (convo[j].isalnum() or (convo[j].isspace()) or (convo[j] in string.punctuation) or convo[j] == "’" or convo[j] == "‘"):
                base.append(convo[j])
                base_joined = ''.join(base) #combine chars
                clear_withs  = base_joined.replace("w/", "with") #fix withs
                post_index = clear_withs[0]
                is_written = False
                no_index = clear_withs.lstrip('0123456789.') #remove the index numbers of the posts
                no_tags = re.sub('(?<=^|(?<=[^a-zA-Z0-9-_\.]))#([A-Za-z]+[A-Za-z0-9-_]+)', '', no_index)
                no_mentions = re.sub('(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)', '', no_tags)
                single_punctuaions = re.sub(r'[\?\.\!]+(?=[\?\.\!])', '', no_mentions)

                abrevs = re.findall(r"\b[A-Z]{2,}\b", single_punctuaions)
                for abrev in abrevs:
                    single_punctuaions = single_punctuaions.replace(abrev, abrev.title())

        if (post_index in conversations[i][2] and not is_written):
            hateful.write(f"{single_punctuaions} \n")
            is_written = True
        else:
            non_hateful.write(f"{single_punctuaions} \n")

non_hateful.close()
hateful.close()



    