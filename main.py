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

for i in range(len(conversations)): #each conversation
    for convo in conversations[i][1].split('\n'): #each post in the conversation
        convo = re.sub(r'http\S+', '', convo) #remove links
        base = []

        for j in range(len(convo)): #for each char in a post
            if (convo[j].isalnum() or (convo[j].isspace()) or (convo[j] in string.punctuation) or convo[j] == "’" or convo[j] == "‘"):
                base.append(convo[j])
                
        base_joined = ''.join(base) #combine chars
        clear_withs  = base_joined.replace("w/", "with") #fix withs
        post_index = clear_withs[:1]
        is_written = False
        no_index = clear_withs.lstrip('0123456789.') #remove the index numbers of the posts
        no_tags = re.sub('(?<=^|(?<=[^a-zA-Z0-9-_\.]))#([A-Za-z0-9]+[A-Za-z0-9-_]+)', '', no_index)
        no_mentions = re.sub('(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z0-9]+[A-Za-z0-9-_]+)', '', no_tags)
        single_punctuaions = re.sub(r'[\?\.\!\:\;\<\>\(\)\*]+(?=[\?\.\!\:\;\<\>\(\)\*])', '', no_mentions)
        no_tabs = single_punctuaions.replace("\t", "") #remove the tabs from the beginning of the post
        no_space_before_punc = re.sub(r'\s+([?.!:;,])', r'\1', no_tabs) #remove spaces before certain punctiations
        no_punc_emojis = re.sub(r'[\?\.\!\:\;\(\)\-]{2,}', '', no_space_before_punc) #remove emojis made with punctiation characters
        space_after_punc = re.sub(r'(?<=[.,!?:;&])(?=[^\s])', r' ', no_punc_emojis) #add space after certain punctiations
        no_multiple_spaces = ' '.join(space_after_punc.split()) #reduce multiple whitespaces to single one
        first_let_capizalized = re.sub("(^|[.?!])\s*([a-zA-Z])", lambda p: p.group(0).upper(), no_multiple_spaces) #capitalize the words at the start of a sentence

        abrevs = re.findall(r"\b[A-Z]{2,}\b", first_let_capizalized)
        for abrev in abrevs:
            first_let_capizalized = first_let_capizalized.replace(abrev, abrev.title())


        if (first_let_capizalized.strip() == ''):
            pass

        elif (post_index in conversations[i][2] and not is_written):
            hateful.write(f"{first_let_capizalized} \n")
            is_written = True

        elif (post_index not in conversations[i][2] and not is_written):
            non_hateful.write(f"{first_let_capizalized} \n")
            is_written = True


non_hateful.close()
hateful.close()