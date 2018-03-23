# Identify specific Incidents as being incorrectly classified as service center
#     Get predictions of more specific Business Services
# Developed by Nathan Shepherd
 
print('Importing dependencies ...')
import warnings # Prevents Gensim from printing an error on EVERY RUN
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=UserWarning, module='ipykernel_launcher')

from gensim import corpora, models, similarities
from nltk import word_tokenize, sent_tokenize
from stopwords import stopwords
import pandas as pd
import numpy as np
import pickle

import codecs
import random
import re
import os

'''####################################'''
'''          Preprocess Data           '''

print('\nImporting training and testing data ...')
file = codecs.open('./All_Reports/business_service_training_data.csv', "r",encoding='utf-8', errors='ignore')
full_report = pd.read_csv(file)
file.close()

for field in full_report:
    full_report[field].fillna('Null', inplace=True)
    
training_data = {'number':full_report['number'].tolist(),
                 'sho_des':full_report['short_description'].tolist(),
                 'description':full_report['description'].tolist(),
                 'service_detail':full_report['u_service_detail'].tolist(),
                 'business_service':full_report['u_business_service'].tolist()}

file = codecs.open('./All_Reports/service_center_compiled.csv', "r",encoding='utf-8', errors='ignore')
serv_cent = pd.read_csv(file)
file.close();#print(serv_cent.head())

for field in serv_cent:
    serv_cent[field].fillna('Null', inplace=True)

testing_data = {'number':serv_cent['number'].tolist(),
                'sho_des':serv_cent['short_description'].tolist(),
                'description':serv_cent['description'].tolist(),
                'service_detail':serv_cent['u_service_detail.name'].tolist(),
                'business_service':serv_cent['u_business_service'].tolist()}

#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#

length_train = len(full_report['number'].tolist())#~176000
print('--> Rows in training data:',length_train)

length_test = len(serv_cent['number'].tolist())#~29000
print('--> Rows in service_center_compiled:',length_test)

try:
    print('\nAttempting to import auxiliary dependencies ...')
    print('--> Loading Corpora ...')
    train_corpus = pickle.load( open( "train_corpus.pkl", "rb" ) )
    test_corpus = pickle.load( open( "test_corpus.pkl", "rb" ) )

    print('--> Loading Word2Vec model ...')
    model = models.Word2Vec.load('model1_shape-100')
    x_shape = 250 #number of dimensions per word
    

except Exception as Execution_of_the_following_code:
    print('\n',Execution_of_the_following_code,'\n')
    print('\tIMPORT FAILED')

    def process_corpus(corpus, field):
        raw = full_report[field]
        outs = []

        #FIXME: recover /keep sentence structure
        for inc in raw:
            #word_len += len(word_tokenize(inc))
            #find and replace all non letters and integers to blank space
            clean = re.sub("[^a-zA-Z?@]", " ", str(inc))

            inc = word_tokenize(clean.lower())#method .lower() decreases dictionary size by 15%

            for word in inc:
                if word in stopwords:
                    inc.remove(word)

            outs.append(inc)

        corpus[field] = outs

    train_corpus = {'description':[],
                    'short_description':[]}
    test_corpus = {'description':[],
                   'short_description':[]}

    train_corpus['description'] = full_report['description'].tolist()
    train_corpus['short_description'] = full_report['short_description'].tolist()

    test_corpus['description'] = serv_cent['description'].tolist()
    test_corpus['short_description'] = serv_cent['short_description'].tolist()

    print('\nProcessing text, this may take awhile ...')
    for field in train_corpus:
        print('--> Processing {} in train_corpus'.format(field))
        process_corpus(train_corpus, field)

    for field in test_corpus:
        print('--> Processing {} in test_corpus'.format(field))
        process_corpus(test_corpus, field)

    pickle.dump( train_corpus, open( "train_corpus.pkl", "wb" ))
    pickle.dump( test_corpus, open( "test_corpus.pkl", "wb" ))

    '''
    Word2Vec curators have semantic analysis simple
    gensim (word2vec)© Copyright 2009-now, Radim Řehůřek
    http://www.fi.muni.cz/usr/sojka/papers/lrec2010-rehurek-sojka.pdf
    ''' 
    feed_model = train_corpus['description'] + train_corpus['short_description']
    feed_model += test_corpus['description'] + test_corpus['short_description']
    tokenized_corp = [inc for inc in feed_model]

    x_shape = 100#@15, ~63%: @100, ~69%, @200, ~80%
    model = models.Word2Vec(tokenized_corp, min_count=1, size=x_shape)
    model.save('model1_shape-100')
    
    ########
    this_word = ('printing')
    print('\nRelative words to \'{}\'\n'.format(this_word))
    for pair in model.most_similar(this_word):
        print(pair)

#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#
# Convert all text into matricies
# Store all matricies as Incident objects

class Incident():
    lexicon = [] #Array representing input (unique matricies in model)
    def __init__(self, number, sho_des, description, service_detail, business_service):
        self.number = number
        self.sho_des = sho_des
        self.description = description
        self.service_detail = service_detail
        self.business_service = business_service
        
        self.database = {'':[self.number],
                         '===Service Detail':service_detail,
                         '===Business Service':business_service,
                         '===Short Description':sho_des,
                         '===Description':description}
    def show(self):
        for field in self.database:
            print(field, self.database[field])

print('\nCreating Incident objects ...')
all_incidents = []
for i in range(length_train):
    all_incidents.append(Incident(training_data['number'][i],
                                  train_corpus['description'][i],
                                  training_data['service_detail'][i],
                                  train_corpus['short_description'][i],
                                  training_data['business_service'][i]))
pickle.dump( all_incidents, open( "other_business_services.pkl", "wb" ))

serv_cent_incs = []
for i in range(length_test):
    serv_cent_incs.append(Incident(testing_data['number'][i],
                                   test_corpus['description'][i],
                                   testing_data['service_detail'][i],
                                   test_corpus['short_description'][i],
                                   testing_data['business_service'][i]))

#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#    
# Use bag of words model to store each matrix by business service

print('\n#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#\n')
print('Correlating all matricies to associated business service')

alpha = 'fghjkvwxyz'
stops = [' ','','a','b','c','d','e','f','g','h','j','k','l','m','n','m','o','p','q','r','s','t','u','v','x','y','z']

def configure(inc):
    x_length = 25# max: 50
    inc.lexicon = []
    for word in inc.sho_des:
        inc.lexicon.append(word)
    for word in inc.description:
        inc.lexicon.append(word)
        
    if len(inc.lexicon) > x_length:
        inc.lexicon = inc.lexicon[:x_length]
        
    if len(inc.lexicon) <= x_length:
        len_append = x_length - len(inc.lexicon)
        padding = []
        for i in range(len_append):
            padding.append(alpha[random.randint(0, len(alpha)-1)])

        if len_append > 0:
            for char in padding:
                inc.lexicon.append(char)
    
    for word in range(0, len(inc.lexicon)):
        this = inc.lexicon[word].lower()
        try:
            inc.lexicon[word] = model[this]#became matrix
        except:
            #print('Failed to model {}'.format(this))
            inc.lexicon[word] = model[alpha[random.randint(0, len(alpha)-1)]]
    #inc.lexicon = np.reshape(inc.lexicon, -1) #1d vector:(250* len(inc.lexicon))

print('\n--> Configuring Incident Lexicons ...')
##try:
##    for i in range(0, 10):
##        all_incidents[i : i+1] = pickle.load(open('all_incidents_configured{}'.format(i), 'rb'))
##    serv_cent_incs = pickle.load(open('serv_cent_configured', 'rb'))

#except Exception as e:
#print(e)
for i, inc in enumerate(all_incidents):
    if i % 25000 == 0:
        print('--<> {}% of all_incidents configured'.format((i*100)//len(all_incidents)))
    configure(inc)
for inc in serv_cent_incs:
    configure(inc)

##    for i in range(0,10):
##        pickle.dump(all_incidents[i : i+1], open('all_incidents_configured{}'.format(i), 'wb'))
##    pickle.dump(serv_cent_incs, open('serv_cent_configured', 'wb'))
    

# Use database and cosine similarities to get predictions
#  -> Find some metric for individual business services
#  -> Find which service_center Incidents are incorrectly classified
#  ---<> Use max cosine similarity as predictor
























































