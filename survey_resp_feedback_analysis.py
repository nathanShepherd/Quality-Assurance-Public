# Get meaningful human emotion from text
# # #
# Developed by Nathan Shepherd
#     with help from Siraj Raval
# --> https://youtu.be/o_OZdbCzHUA?t=2m54s
 
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
stops = ["a",'n', "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the", "n"]
from gensim import corpora, models, similarities
from textblob import TextBlob
import pandas as pd
import re
import codecs
import random
from nltk import word_tokenize
import numpy as np
import matplotlib.pyplot as plt
random.seed(10)

file = codecs.open('Input/input_data.csv', "r",encoding='utf-8', errors='ignore')
q_file = codecs.open('Input/Feedback_Detail_response_questions.csv', "r",encoding='utf-8', errors='ignore')
df = pd.read_csv(file)
questions = pd.read_csv(q_file)
q_file.close()
file.close()

inc_nums = df['Number'].tolist()
length = len(df['Number'].tolist())
responses = df['Response'].tolist()

out_data = {'Emotional Values':[],'Emotional Values Text':[]}
subj_dict = {'Degree of Objectivity':[],'Degree of Objectivity Text':[]}

### Analyze question feedback and get interval that includes 95% of all negative responses

for header in questions:
    questions[header].fillna('5', inplace=True)

#print(questions.head())
q_nums = questions['Unnamed: 0'].tolist()[1:]

q1 = [int(q) for q in questions['QUESTION.1'].tolist()[1:]]
q2 = [int(q) for q in questions['QUESTION.2'].tolist()[1:]]
q3 = [int(q) for q in questions['QUESTION.3'].tolist()[1:]]
q4 = [int(q) for q in questions['QUESTION.4'].tolist()[1:]]

total = [0 for i in range(11)]
for i in q1:
    total[i] += 1
for i in q2:
    total[i] += 1
for i in q3:
    total[i] += 1
for i in q4:
    total[i] += 1

#plt.bar(range(10), total[:-1])
#plt.show()
#total -> [0, 328, 47, 46, 38, 113, 52, 75, 145, 464, 11952]

for i in range(len(inc_nums)):
    found = False
    for q_i in range(len(q_nums)):
        if inc_nums[i] == q_nums[q_i] and not found:
            found = True
            resp = (q1[q_i] + q2[q_i] + q3[q_i] + q4[q_i])/4
            #print(resp)
            
            this = TextBlob(responses[i])
            out_data['Emotional Values'].append(str(this.sentiment.polarity))
            subj_dict['Degree of Objectivity'].append(str(this.sentiment.subjectivity))
            if this.sentiment.subjectivity > 0.5:
                subj_dict['Degree of Objectivity Text'].append('Subjective')
            else:
                subj_dict['Degree of Objectivity Text'].append('Objective')
            if this.sentiment.polarity > -0.1:
                out_data['Emotional Values Text'].append('Positive')
            else:
                out_data['Emotional Values Text'].append('Negative')
                    
            bigram = []
            all_words = word_tokenize(responses[i].lower())
            for i in range(len(all_words) - 1):
                bigram.append(all_words[i] +' '+ all_words[i + 1])
                
            if float(out_data['Emotional Values'][-1]) < 0.1 and float(out_data['Emotional Values'][-1]) > -0.25:
                if resp == 10:
                    out_data['Emotional Values'][-1] = '4'
                    out_data['Emotional Values Text'][-1] = 'Positive'

                elif resp == 1 and 'thank you' not in bigram:
                    out_data['Emotional Values'][-1] = '-2'
                    out_data['Emotional Values Text'][-1] = 'Negative'
                
    if not found:
        print('Failed to find Incident in questions file:',inc_nums[i])
            

'''
for i in range(length):
    this = TextBlob(responses[i])
    out_data['Emotional Values'].append(str(this.sentiment.polarity))
    if this.sentiment.polarity > -0.1:
        out_data['Emotional Values Text'].append('Positive')
    else:
        out_data['Emotional Values Text'].append('Negative')
'''
out_data = pd.DataFrame(out_data)

#pause = input("\nWaiting . . .")

df = pd.concat([df, out_data], axis=1)
df = pd.concat([df, pd.DataFrame(subj_dict)], axis=1)
df.to_csv('Analyzed_feedback_emotional_value.csv')
print(df.head())


########################################################
print('\n\nPlease verify the data according to the rules in Step 7.\nPress enter to continue.')
#wait_for = input()
print('Make sure Analyzed_feedback_emotional_value.xlsx is closed and press enter when Step 7 is completed.')
#wait_for = input()

#Get meaningful human emotion from text
# # #
# Developed by Nathan Shepherd
#     with help from Siraj Raval
# --> https://youtu.be/o_OZdbCzHUA?t=2m54s

from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import pandas as pd
import operator
import codecs
import nltk

file = codecs.open('Analyzed_feedback_emotional_value.csv', "r",encoding='utf-8', errors='ignore')

df = pd.read_csv(file)
file.close()

length = len(df['Number'].tolist())
responses = df['Response'].tolist()
polarity = df['Emotional Values Text'].tolist()

out_data = {'Positive Classification':['None' for i in range(length)],
            'Negative Classification':['None' for i in range(length)]}

descriptive_adjectives = ['thank','helpful','patient','appreciate','knowledgeable',
                          'friendly','professional','quickly','appreciated','quick',
                          'courteous','thank','excellent','thanks','nice','pleasant',
                          'kind','excellent','fast','awesome','good','fix','fixed',
                          'super','patience','polite','wonderful','amazing',
                          'outstanding','appreciate','timely','grateful','explained',
                          'fantastic','happy','satisfied','terrific','impressed',
                          'quick','personable','exceptional','super','effective',
                          'professionalism','knowledgable','willingness','informative',
                          'efficiently','thankful','willing','appreciative','amazingly',
                          'patiently','Appreciated','reassuring','exemplary','exceptionally',
                          'quality','fabulous','Friendly','perfectly','professionally',
                          'great','help','best','service','greatest','love','enjoyed',
                          'easy','clear','thoughtful','glad','superb','loved','thanx',
                          'efficient','pleasure','satisfactory','Thank-you','courteously',
                          'efficiency','thoughtful','skillful','confident','flexible',
                          'enthusiasm','effectively','persistence','promptness','friendliness',
                          'extraordinarily','confidence','kindness','assistance','helped','understanding']

negative_descriptors = ['issue','wrong','closed','issues','disappointed','bad','problems',
                        'confused','terrible','ridiculous','failed','hated','reluctant',
                        'unappreciative','unhappy','headache','irritating', 'problems',
                        'problem','failure','poor', 'reasoning','waste', 'sucks',
                        'helped','difficult','difficulty', 
                        'complain', 'regret','outdated','happy','helped','closed','solved',
                        'long','ridiculous','excessive','redirected','educated',
                        'fixed','hard','terrible','resolved','expired','hated',
                        'quality','unhappy','issues','unappreciative',
                        'clumsy','unnecessarily', 'complicated','irritating','persistant',
                        'issue','long','time','disappointed','DuoToken','annoying',
                        'ridiculous','wait','resolved','headache','disappointed']
for i in range(length):
    this = responses[i]
    if polarity[i] == 'Positive':
        this = word_tokenize(str(responses[i]))
        container = []
        matched = False
        for word in this:
            for syn in descriptive_adjectives:
                if word.lower() == syn.lower():
                    matched = True
                    if word not in container:
                        container.append(word)
        if matched:
            string = ''
            for word in container:
                string = string +' '+ word
            out_data['Positive Classification'][i] = string

    elif polarity[i] == 'Negative':
        this = word_tokenize(str(responses[i]))
        container = []
        matched = False
        for word in this:
            for syn in negative_descriptors:
                if word.lower() == syn.lower():
                    matched = True
                    if word not in container:
                        container.append(word)
        if matched:
            string = ''
            for word in container:
                string = string +' '+ word
            out_data['Negative Classification'][i] = string

out_data = pd.DataFrame(out_data)

df = pd.concat([df, out_data], axis=1)
print(df.head)
df.to_csv('Analyzed_feedback_emotional_value.csv')

##########################################################################

import codecs
import nltk

#file = codecs.open('./Input/Analyzed_feedback_emotional_value_early_july27.csv', "r",encoding='utf-8', errors='ignore')
file = codecs.open('Analyzed_feedback_emotional_value.csv', "r",encoding='utf-8', errors='ignore')

df = pd.read_csv(file)
file.close()

length = len(df['Number'].tolist())
responses = [str(resp).lower() for resp in df['Response'].tolist()]
polarity = df['Emotional Values Text'].tolist()

pos_class = df['Positive Classification'].tolist()
neg_class = df['Negative Classification'].tolist()

out_data = {'Positive Classification':['None' for i in range(length)],
            'Negative Classification':['None' for i in range(length)]}



############
'''
Simplify featureset
'''
############
concats = {'Thankful':['Thank','Thanks','you','thanks',
                       'thank','THANK','THANKS','thankful','Thank-you',
                       'Thanx','thank-you',],
           #
           'Helpful':['helpful','help','helped','assistance','easy',
                      'fix','fixed','Solved','explained','Help','love',
                      'Love','Helpful','loved','thoughtful','impressed',
                      'HELPFUL','Amazingly','hopeful','Clear','inquiery',
                      'Loved','super-helpful','Fix','FLEXIBLE','helpfulness',
                      'skillful','enjoyed','Fixed','persistence','resolutoin',
                      'informative','willingness','willing','HELP','Friendly',
                      'Easy','effectively','Helped','Assistance','Exceptional',],
           #
           'Professional':['professionalism','efficiently','AMAZING',
                           'Amazing','superb','perfectly','fabulous',
                           'professionally','Outstanding','exemplary',
                           'Professional','professional','nice','kind',
                           'polite','amazing','pleasure','outstanding',
                           'clear','terrific','personable','exceptional',
                           'Terrific','quality','FANTASTIC','confidence',
                           'POLITE','OUTSTANDING','WONDERFUL','wonderfull',
                           'kindness','confident','satisfactory','friendliness',],
                           
            #more definite
           #
           'Patient':['patient','patience','patiently','PATIENT','Patient',],
           #
           'Friendly':['friendly','pleasant',],
           #
           'Courteous':['courteous','Courteous','courtesy','courteously',],
           #
           'Knowledgeable':['knowlegable','Knowledgeable','knowledgable','knowledgeable','knowledge',],
           
           'Timely / Efficient':['quickly','quick','efficient','fast','timely','Quick','Fast',
                                 'efficiency','effective','promptness','flexible','prompt','Quickly',
                                 'Efficient','test','thorough','resolution','done',
                                 'FAST','speedy',],
           #
           'Appreciative':['service','great','appreciate',
                           'Great','appreciated','excellent',
                           'Excellent','awesome','good','best',
                           'super','wonderful','Service','grateful',
                           'fantastic','happy','Appreciate','Awesome',
                           'satisfied','Good','Best','Super','GREAT',
                           'glad','AWESOME','appreciative','Fantastic',
                           'amazingly','EXCELLENT','Wonderful','Appreciated',
                           'reassuring','exceptionally','Nice','BEST','SUPER',
                           'extraordinarily','Happy','enthusiasm','Pleasant',
                           'GOOD','Glad','greatest','Satisfied','Grateful',
                           'SUPERB','HAPPY','very','KIND','Fabulous','persistant',
                           'APPRECIATE','SERVICE','SATISFIED','GRATEFUL',
                           'PLEASURE','well','NICE','GLAD','Superb','FABULOUS','understanding',],
           'None':['None']}

neg_cats = {'Problem or Issue':['problem','issue','closed','issues','persistant',
                               'fixed','problems','Problem','unnecessarily','solved'],
            
            'Too Slow':['time','long','expired','slow','slowly','TIME','redirect',],
            
            'Poor Service':['closed','wrong','unresolved','disappointed','wait',
                            'bad','difficult','poor','hard','failed','excessive',
                            'confused','ridiculous','terrible','process','SUCKS',
                            'unappreciative','waste','disliked','failure','frustrated',
                            'regret','unsatisfactory','WASTE','unhappy','not-happy',
                            'reasoning','quality','outdated','reluctant','hated','headache',
                            'annoying','complicated','educated','DuoToken','Resolved','complain',
                            'sucks','irritating'],
            'None':['None']}

for i, row in enumerate(neg_class):
    tokenized = word_tokenize(row)
    negatives = {'Problem or Issue':0,
                 'Too Slow':0,
                 'Poor Service':0,
                 'None':0}
    for word in tokenized:
        for category in neg_cats:
            for syn in neg_cats[category]:
                if category == 'Problem or Issue':
                    if word.lower() == syn.lower():
                        negatives['Problem or Issue'] += 1
                if category == 'Too Slow':
                    if word.lower() == syn.lower():
                        negatives['Too Slow'] += 1
                if category == 'Poor Service':
                    if word.lower() == syn.lower():
                        negatives['Poor Service'] += 1
                if category == 'None':
                    if word.lower() == syn.lower():
                        negatives['None'] += 1
    classifier = ''
    _max = 0
    for key in negatives:
        if negatives[key] > _max:
            _max = negatives[key]
            classifier = key
    #print('\n',tokenized)
    #print('--->',classifier, '\n')
    if classifier == '':
        classifier = 'Poor Service'
    out_data['Negative Classification'][i] = classifier
    
dictionary = {}
for i, row in enumerate(pos_class):
    tokenized = word_tokenize(row)
    cats = {'Timely / Efficient':0,
            'Professional':0,
            'Appreciative':0,

            'Patient':0,
            'Friendly':0,
            'Courteous':0,
            'Knowledgeable':0,
            
            'Thankful':0,
            'Helpful':0,
            'None':0}
    for word in tokenized:
        if word not in dictionary:
            dictionary[word] = 0
        else:
            dictionary[word] += 1
        for category in concats:
            for syn in concats[category]:
                if category == 'Thankful':
                    if word.lower() == syn.lower():
                        cats['Thankful'] += 1
                if category == 'Helpful':
                    if word.lower() == syn.lower():
                        cats['Helpful'] += 1
                if category == 'Professional':
                    if word.lower() == syn.lower():
                        cats['Professional'] += 1
                        
                if category == 'Timely / Efficient':
                    if word.lower() == syn.lower():
                        cats['Timely / Efficient'] += 1
                if category == 'Appreciative':
                    if word.lower() == syn.lower():
                        cats['Appreciative'] += 1
                if category == 'None':
                    if word.lower() == syn.lower():
                        cats['None'] += 1

                if category == 'Patient':
                    if word.lower() == syn.lower():
                        cats['Patient'] += 1
                if category == 'Friendly':
                    if word.lower() == syn.lower():
                        cats['Friendly'] += 1
                if category == 'Courteous':
                    if word.lower() == syn.lower():
                        cats['Courteous'] += 1
                if category == 'Knowledgeable':
                    if word.lower() == syn.lower():
                        cats['Knowledgeable'] += 1
    classifier = ''
    _max = 0
    for key in cats:
        if cats[key] > _max:
            _max = cats[key]
            classifier = key
    #print('\n',tokenized)
    #print('--->',classifier, '\n')
    out_data['Positive Classification'][i] = classifier

out_data = pd.DataFrame(out_data)

df = pd.concat([df, out_data], axis=1)
print(df.head)
#df.to_excel('./Output/Analyzed_feedback_emotional_value.xlsx')


#access more response data to improve word vectors
'''
all_data = ''
file = codecs.open('Input/four_years_data.csv', "r",encoding='utf-8', errors='ignore')
df = pd.read_csv(file)
all_data = df['Response'].tolist()
file.close()

for resp in all_data:
    for word in word_tokenize(str(resp)):
        if word not in stops:
            tokenized_corp.append(word.lower())
'''
# how_pos tpo contain = {[   (proffessional empathetic, patient), (proffessional empathetic, patient)]}
# how_neg tpo contain = {[not --> (proffessional empathetic, patient)]}


how_pos = {'Professional':['' for i in range(length)],
           'Empathetic':['' for i in range(length)],
           'Patient':['' for i in range(length)]}

what_pos = {'Helpful':['' for i in range(length)],
            'Efficient':['' for i in range(length)],
            'Knowledgeable':['' for i in range(length)],
            'Positive Non-Specific':['' for i in range(length)]}

how_neg = {'Not Professional':['' for i in range(length)],
           'Not Empathetic':['' for i in range(length)],
           'Not Patient':['' for i in range(length)]}

what_neg = {'Not Helpful':['' for i in range(length)],
            'Not Efficient':['' for i in range(length)],
            'Not Knowledgeable':['' for i in range(length)],
            'Negative Non-Specific':['' for i in range(length)]}

                #false positives with "the kind" and "kind of"
positive_dict = {'Empathetic':['empathetic', 'sympathetic', 'understanding' 'understood',
                               'understood', 'urgency', 'willingness', 'willing',
                               'personable','kind', 'pleasant', 'friendly'],

                 'Patient': ['patient', 'patience', 'patiently'],

                 'Professional':['professionalism','professionally','Professional','POLITE',
                                 'persistence','persistent','thorough','follow-up',
                                 'courteous','courtesy','courteously'],

                 'Efficient':['quickly','quick','efficient','fast','timely','Quick','efficiency',
                              'effective','promptness','flexible','prompt','thorough',
                              'speedy','promptly'],
                 
                 'Helpful':['helpful','help','helped','assistance','easy','fix','fixed','solved',
                            'explained','super-helpful','flexible','resolved','informative'],

                 #checkout more specific synonyms for skills and knowledge
                 'Knowledgeable':['knowledgeable','skillful','skills','expertise','Knowledge',
                                  'expert','skilled','experienced','savvy','brilliant'],

                 'Positive Non-Specific':['Thank','Thanks','you','thankful','Thank-you','Thanx',
                                           'service','great','appreciate','Great','appreciated',
                                           'excellent','Excellent','awesome','good','best','super',
                                           'wonderful','Service','grateful','fantastic','happy','satisfied',
                                           'glad','appreciative','amazingly','reassuring','exceptionally',
                                           'Nice','extraordinarily','Happy','enthusiasm','Pleasant','greatest',
                                           'Grateful','SUPERB','Fabulous','well',]
                 }
pos_class = [resp for resp in responses if polarity[i] == 'Positive']
for i, row in enumerate(pos_class):
    tokenized = word_tokenize(row)
    cats = {'Patient':0,
            'Helpful':0,
            'Efficient':0,
            'Empathetic':0,
            'Professional':0,
            'Knowledgeable':0,
            'Positive Non-Specific':0}
    
    for j, word in enumerate(tokenized):
        if word not in dictionary:
            dictionary[word] = 0
        else:
            dictionary[word] += 1
        for category in positive_dict:
            for syn in positive_dict[category]:
                if category == 'Empathetic':
                    if word.lower() == syn.lower():
                        if word == "kind" and j < len(tokenized) - 1:
                            if (tokenized[j+1] != "of" and tokenized[j-1] != "the"):
                                cats['Empathetic'] += 1
                        else:
                            cats['Empathetic'] += 1
                if category == 'Patient':
                    if word.lower() == syn.lower():
                        cats['Patient'] += 1
                if category == 'Helpful':
                    if word.lower() == syn.lower():
                        cats['Helpful'] += 1
                        
                if category == 'Efficient':
                    if word.lower() == syn.lower():
                        if word == 'efficient' and j < len(tokenized) - 1:
                            if tokenized[j + 1] != "way" and tokenized[j - 1] != "more":
                                cats['Efficient'] += 1
                        elif inc_nums[i] in ['INC1085117','INC1072342','INC1038377','INC1003620']:
                            pass
                        else:
                            cats['Efficient'] += 1
                if category == 'Professional':
                    if word.lower() == syn.lower():
                        cats['Professional'] += 1
                if category == 'Knowledgeable':
                    if word.lower() == syn.lower():
                        cats['Knowledgeable'] += 1
                if category == 'Positive Non-Specific':
                    if word.lower() == syn.lower():
                        cats['Positive Non-Specific'] += 1
        
    #Identify and populate positive 'How' dictionary
    if (cats['Professional'] >= 1):
        how_pos['Professional'][i] = 1
    else:
        how_pos['Professional'][i] = 0
        
    if (cats['Empathetic'] >= 1):
        how_pos['Empathetic'][i] = 1
    else:
        how_pos['Empathetic'][i] = 0
        
    if (cats['Patient'] >= 1):
        how_pos['Patient'][i] = 1
    else:
        how_pos['Patient'][i] = 0

    #Identify and populate positive 'What' dictionary
    if (cats['Helpful'] >= 1):
        what_pos['Helpful'][i] = 1
    else:
        what_pos['Helpful'][i] = 0
        
    if (cats['Efficient'] >= 1):
        what_pos['Efficient'][i] = 1
    else:
        what_pos['Efficient'][i] = 0
        
    if (cats['Knowledgeable'] >= 1):
        what_pos['Knowledgeable'][i] = 1
    else:
        what_pos['Knowledgeable'][i] = 0

    if cats['Positive Non-Specific'] >= 1:
        what_pos['Positive Non-Specific'][i] = 1
        what_neg['Negative Non-Specific'][i] = 0
    else:
        what_pos['Positive Non-Specific'][i] = 0
        what_neg['Negative Non-Specific'][i] = 0

    #check if entire row is empty
    if (how_pos['Patient'][i] == 0 and what_pos['Helpful'][i] == 0 and
        how_pos['Professional'][i] == 0 and how_pos['Empathetic'][i] == 0 and
        what_pos['Efficient'][i] == 0 and what_pos['Knowledgeable'][i] == 0 and
        polarity[i] == 'Positive'):
        
        what_pos['Positive Non-Specific'][i] = 1
        what_neg['Negative Non-Specific'][i] = 0
    else:
        what_pos['Positive Non-Specific'][i] = 0
        what_neg['Negative Non-Specific'][i] = 0

negative_dict = {'Not Empathetic':['understanding','hated', 'frustrated'],#'rude','unwilling','unwillingness','understand',

                 #looking for incidents that have the customer hurry up and go away
                 'Not Patient':[''],#'impatient','patience','rushed','time',

                 #identify why none of these where identified,'ignorant','incapable','inexperienced',
                 #'professionalism','Professional','incompetent','unprofessional'
                 
                 # look for sequences of two words (too much), bad impression,  please dont tell me 
                 'Not Professional':['annoying','terrible','unsatisfactory',#'wrong','failed',
                                     'dont tell me','communication','too much',#'too', 'much',
                                     'bad impression'],

                 #consider "took", "associates", 'knowlegable','know','knowledge','informative','skill'
                                     
                 #trigram?: figure out myself, bigram: not done, resolved problem myself, no solution, thinking
                 #I would have expected mreo experirence, didn;t know much, difficult troubleshooting, not knowldege
                 'Not Knowledgeable':['persistant','educated','understanding','information','communication',
                                      'figure out myself',#'poor','problem','workaround','know','multiple'
                                      'not done','resolved problem myself','thinking'],#'no solution'

                 #back and forth, forwarded, several months, multiple visits, call back, took a week
                 #Check over a few: "had a difficult time", "include specific contexts of 'time'"
                 'Not Efficient':['long', 'expired','slow','redirect','minutes','multiple visits',
                                  'hold','minute','forwarded','took a week','call back','several months',
                                  'forwarded','back and forth','five weeks'],

                 #bi gram: work around, several attemps, did not work, if your only advice, didn't work,
                                    #didnt adress the problem, problem, not returned a phone call
                 'Not Helpful':['unhappy','resolved','quality','closed','work around','communication','wait'
                                'problem','solved','disappointed','help','didnt work','your only advice',
                                'did not work','several attemps','work around']}#'unhelpful','completed'

for i, resp in enumerate(responses):
    if polarity[i] == 'Negative':
        resp = resp.replace('\'', '')
        tokenized = word_tokenize(resp)
        bigram = [tokenized[i] +' '+ tokenized[i+1] for i in range(len(tokenized) - 1)]
        trigram = [tokenized[i] +' '+ tokenized[i+1] +' '+ tokenized[i+2] for i in range(len(tokenized) - 2)]
        for phrase in bigram:
            tokenized.append(phrase)
        for phrase in trigram:
            tokenized.append(phrase)
        negatives = {'Not Knowledgeable':0,
                 'Not Professional':0,
                 'Not Empathetic':0,
                 'Not Efficient':0,
                 'Not Helpful':0,
                 'Not Patient':0}
        for word in tokenized:
            for category in negative_dict:
                for syn in negative_dict[category]:
                    if category == 'Not Knowledgeable':
                        if word.lower() == syn.lower():
                            negatives['Not Knowledgeable'] += 1
                    if category == 'Not Professional':
                        if word.lower() == syn.lower() and word.lower not "wrong":
                            negatives['Not Professional'] += 1
                    if category == 'Not Empathetic':
                        if word.lower() == syn.lower():
                            negatives['Not Empathetic'] += 1
                    if category == 'Not Helpful':
                        if word.lower() == syn.lower():
                            negatives['Not Helpful'] += 1
                    if category == 'Not Patient':
                        if word.lower() == syn.lower():
                            negatives['Not Patient'] += 1
                    if category == 'Not Efficient':
                        if word.lower() == syn.lower():
                            negatives['Not Efficient'] += 1


    #Identify and populate negative 'How' dictionary
        if (negatives['Not Professional'] >= 1): #identify why no output
            how_neg['Not Professional'][i] = 1
        else:
            how_neg['Not Professional'][i] = 0
        
        if (negatives['Not Empathetic'] >= 1):
            how_neg['Not Empathetic'][i] = 1
        else:
            how_neg['Not Empathetic'][i] = 0
        
        if (negatives['Not Patient'] >= 1):
            how_neg['Not Patient'][i] = 1
        else:
            how_neg['Not Patient'][i] = 0

        #Identify and populate positive 'What' dictionary
        if (negatives['Not Helpful'] >= 1):
            what_neg['Not Helpful'][i] = 1
        else:
            what_neg['Not Helpful'][i] = 0
        
        if (negatives['Not Efficient'] >= 1):
            what_neg['Not Efficient'][i] = 1
        else:
            what_neg['Not Efficient'][i] = 0
        
        if (negatives['Not Knowledgeable'] >= 1):#identify why not output
            what_neg['Not Knowledgeable'][i] = 1
        else:
            what_neg['Not Knowledgeable'][i] = 0

    #check if entire row is empty
        if (how_neg['Not Patient'][i] == 0 and what_neg['Not Helpful'][i] == 0 and
            how_neg['Not Professional'][i] == 0 and how_neg['Not Empathetic'][i] == 0 and
            what_neg['Not Efficient'][i] == 0 and what_neg['Not Knowledgeable'][i] == 0):
            if (polarity[i] == 'Negative'):
                what_neg['Negative Non-Specific'][i] = 1
        
        else:
            what_neg['Negative Non-Specific'][i] = 0
    else:
        what_neg['Not Knowledgeable'][i] = 0
        how_neg['Not Professional'][i] = 0
        how_neg['Not Empathetic'][i] = 0
        what_neg['Not Efficient'][i] = 0
        what_neg['Not Helpful'][i] = 0
        how_neg['Not Patient'][i] = 0
        
'''
file = codecs.open('Input/responses_2013_2017.csv', "r",encoding='utf-8', errors='ignore')
old_data = pd.read_csv(file)
old_resp = old_data["Response"].tolist()
file.close()
'''
#ensure all positive do not have negative ones and vise versa
for i, resp in enumerate(responses):
    if polarity[i] == 'Negative':
        what_pos['Positive Non-Specific'][i] = 0
        what_pos['Knowledgeable'][i] = 0
        what_pos['Efficient'][i] = 0
        what_pos['Helpful'][i] = 0
        
        how_pos['Professional'][i] = 0
        how_pos['Empathetic'][i] = 0
        how_pos['Patient'][i] = 0
        
    if polarity[i] == 'Positive':
        what_neg['Negative Non-Specific'][i] = 0
        what_neg['Not Knowledgeable'][i] = 0
        what_neg['Not Efficient'][i] = 0
        what_neg['Not Helpful'][i] = 0
        
        how_neg['Not Professional'][i] = 0
        how_neg['Not Empathetic'][i] = 0
        how_neg['Not Patient'][i] = 0
        
        

# Standard Deviation: 116
# Unique Words: 5006
# Mean: 15
freq_dict = {}
for resp in responses:
    all_words = word_tokenize(resp)
    for word in all_words:
        if word not in freq_dict:
            freq_dict[word] = 1
        else:
            freq_dict[word] += 1

# measure average similarity among all words (within sigma)
# identify average deviation of words in all responses
# verify responses outside this range are non-service related
'''
import numpy as np
not_service_related = {'Not Service Related':['' for i in range(length)]}
distance = [1 for i in range(length)]
for word in freq_dict:
    freq_dict[word] = (freq_dict[word])/len(freq_dict)

for i, resp in enumerate(responses):
    all_words = word_tokenize(resp)
    if "survey" in all_words and len(all_words) < 30 and "solved" not in all_words or "offboarding" in all_words:
        not_service_related['Not Service Related'][i] = 1
    else:
        not_service_related['Not Service Related'][i] = 0
    
    for word in all_words:
        distance[i] += freq_dict[word]
        
    if len(all_words) < 75:
        distance[i] = distance[i] / len(all_words)

sigma = np.std(distance)
mu = np.mean(distance)

for i in range(len(distance)):
    distance[i] = (distance[i] - mu)/sigma
'''
count = 0

freq_dict = {}
for i, resp in enumerate(responses):
    if polarity[i] == 'Negative':
        all_words = word_tokenize(resp)
        for word in all_words:
            if word not in freq_dict:
                freq_dict[word] = 1
            else:
                freq_dict[word] += 1

##########################
'''
INC0898000 <- Should be Positive
INC0917773 <- Should be Positive
INC0951090 <- should be positive
'''
##########################


##import matplotlib.pyplot as plt
##histo, bins = np.histogram(distance, bins=200)
##
##fig,ax = plt.subplots()
##ax.bar(bins[:50], histo[:50], width=0.8, color='purple')
##plt.show()

how_pos = pd.DataFrame(how_pos)
df = pd.concat([df, how_pos], axis=1)

how_neg = pd.DataFrame(how_neg)
df = pd.concat([df, how_neg], axis=1)

what_pos = pd.DataFrame(what_pos)
df = pd.concat([df, what_pos], axis=1)

what_neg = pd.DataFrame(what_neg)
df = pd.concat([df, what_neg], axis=1)

'''
not_service_related = pd.DataFrame(not_service_related)
df = pd.concat([df, not_service_related], axis=1)
'''
#df.drop(df.columns[[14, 15]], axis=1, inplace=True)

print(df.head)
df.to_excel('./Output/ones_and_zeros_Dec_6_updated.xlsx')

#remove redundant columns "helpful" column
###less patients
#stricter enforcement of what 
#combine positive and negative classification into "Emotional Classification"

print('\n\n\n==========\nAnalysis Complete: Press any key to exit\n==========\n')

'''
sorted_dict = sorted(dictionary.items(), key=operator.itemgetter(1))
for pair in sorted_dict:
    print('\'{}\','.format(pair[0]),'\n')
#print([pair[0]for pair in sorted_dict])
'''

