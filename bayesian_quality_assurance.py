# Quality Checker
# Developer: Nathan Shepherd

###
'''
Number and Resolution cat in a new file
--> For input
--> And output

Number and other fields seperate file
--> One file res_cat, other inc_cat
--> Place in directory for later FTP
--> Compute statistics on output
'''
###

print('Importing Dependancies ...')
import re
import time
import codecs
import random
import pickle
import operator
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk import word_tokenize, sent_tokenize

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=UserWarning, module='ipykernel_launcher')

from gensim import corpora, models, similarities
initial_time = time.time()

'''####################################'''
'''          Preprocess Data           '''

# Unverified Data
# # #
# all_incidents in input_data.csv will be analysed for symantic content
#
# Based on the text in self.database, each incident will be categorized

print('\nPulling incident data ...\n')
file = codecs.open('./Input/input_data.csv', "r",encoding='utf-8', errors='ignore')
full_report = pd.read_csv(file)


full_report['category'].fillna('Null', inplace=True)
full_report['u_service_detail'].fillna('Null', inplace=True)
full_report['u_business_service'].fillna('Null', inplace=True)
full_report['u_resolution_category'].fillna('Null', inplace=True)
full_report['close_code'].fillna('Solved (Confirmed by Customer)', inplace=True)
file.close()


#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#

num_incs = len(full_report['u_service_detail'].tolist())
print('Number of Incidents to be checked:',num_incs)

corpus = {'description':[],
          'short_description':[],
          'close_notes':[]}

numbers     = full_report['number'].tolist()
res_cats    = full_report['u_resolution_category'].tolist()
categories  = full_report['category'].tolist()
close_codes = full_report['close_code'].tolist()
assigned_to_init = full_report['assigned_to']
service_details  = full_report['u_service_detail'].tolist()
serv_cat_entries = full_report['u_service_catalog_entry'].tolist()
business_services = full_report['u_business_service']
assignment_group_init = full_report['assignment_group']

corpus['description'] = full_report['description'].tolist()
corpus['close_notes'] = full_report['close_notes'].tolist()
corpus['short_description'] = full_report['short_description'].tolist()

stopwords = ["a",'n', "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the", "n"]

########
def process_corpus(field):
    raw = full_report[field]
    outs = []

    #FIXME: recover /keep sentence structure
    for inc in raw:
        #word_len += len(word_tokenize(inc))
        #find and replace all non letters and integers to blank space
        clean = re.sub("[^a-zA-Z?]", " ", str(inc))
        
        inc = word_tokenize(clean.lower())#method .lower() decreases dictionary size by 15%
        
        for word in inc:
            if word in stopwords:
                inc.remove(word)
        
        outs.append(inc)
        
    corpus[field] = outs
    
for field in corpus:
    process_corpus(field)

#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#

class Incident():
    lexicon = [] #Array representing input (unique matricies from model of words)
    def __init__(self, close_notes, sho_des, description, service_detail,
                     number, category, close_code, res_cat, serv_cat_ent,
                     assignment_group, assigned_to, business_service):
        self.number = number
        self.sho_des = sho_des
        self.res_cat = res_cat
        self.category = category
        self.close_code = close_code
        self.assigned_to = assigned_to
        self.close_notes = close_notes
        self.description = description
        self.serv_cat_ent = serv_cat_ent
        if service_detail == 'Software Application ':
            service_detail = 'Software Application'
        self.service_detail = service_detail
        self.business_service = business_service
        self.assignment_group = assignment_group
        self.database = [[self.number],
                         ['===Short Description', sho_des],
                         ['===Close Notes',close_notes],
                         ['===Description',description[:50]]]
    def show(self):
        for field in self.database:
            print(field)
            
testing_incs = []
for i in range(0, num_incs):
    testing_incs.append(Incident(corpus['close_notes'][i], corpus['short_description'][i],
                                 corpus['description'][i], service_details[i],numbers[i], 
                                 categories[i], close_codes[i],res_cats[i],serv_cat_entries[i],
                                 assignment_group_init[i],assigned_to_init[i],business_services[i]))
#testing_incs = testing_incs[:100]                                 
#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#
        
###'''###'''###'''###'''###
# This cleaned data contains Incident objects that have been pre-processed
#
# Every word in self.sho_des, self.close_notes, & self.description
#     has been tokenized using nltk.word_tokenize()

print("Loading Incident Data")
all_incidents = pickle.load(open("all_incs_cleaned", "rb"))# ~58,000

random.shuffle(all_incidents)
#all_incidents = all_incidents[:2000]
#Word2Vec model created from gensim using all_incidents before cleaning
model = models.Word2Vec.load('model3_shape-250')
model = model.wv
x_shape = 250 #number of dimensions per word

print('Succesfully created',len(all_incidents),'Incident objects with these fields:')
print('\n-->\tShort description, Description, Close notes')
print('\n-->\tService Detail, Category, Service Catalog Entry')
print('\n-->\tService Provider, Resolution Catagory, Close Code\n')


#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%v#%#

from scipy import stats
import numpy as np
import operator
import random
import math

class Bayesian_Classifier():
    def __init__(self):
        pass

    def evaluate_model(self):
        num_success = 0
        len_test = len(self.x)
        for i in range(len_test):
            _, pred = self.classify(self.x[i])

            if pred == self.y[i]:
                num_success += 1
                
        accuracy = num_success*100 / len_test
        print('\nAccuracy of classifier: {} %'.format(accuracy))

        for domain in self.print_dict:
            print('Words that occur most frequently: domain',domain,"{}")
            outs = sorted(self.freq_dict[domain].items(), key=operator.itemgetter(1))[::-1]
            for string in outs[:5]:
                print(string)

    def test_accuracy(self, novel_x, novel_y):
        if type(novel_x) != type(list()):
            print("Input DataType must be list of strings")

        num_success = 0
        len_test = len(novel_x)
        for i in range(len_test):
            _, pred = self.classify(novel_x[i])

            if pred == novel_y[i]:
                num_success += 1

        accuracy = num_success*100 / len_test
        print('\nAccuracy of classifier: {} %'.format(accuracy))
            

    def classify(self, input_data):
        # y == domain, yi = classifier[domain]
        #argmax( P(doc | y) * P(y))
        #argmax( P(x1, x2, ...| y) * P(y))
        # --> P(x1,x2,...|y) == P(x1|y)*P(x2|y)* ...
        #Thus, for all y's in y:
        # --> yi = argmax(P(yi) * (P(x1|yi)*P(x2|yi)* ...))
        #(in this case, P(yi) is the same for all y's in yi)

        _max = [0, -1]# [argmax(P(yi)), yi]
        all_products = []

        for domain in self.freq_dict:
            product = 1
            for word in input_data:
                try:
                    product *= self.freq_dict[domain][word]
                except KeyError as e:
                    pass
            
            if product > _max[1]:
                _max = [domain, product]

            all_products.append(product)

        #add to z_score dividend to prevent divide by zero error
        z_score = ((_max[1] - np.mean(all_products)) / (np.std(all_products) + 0.01))
        accuracy_of_prediction = 1 - np.exp(-(z_score ** 2) / 2)#~N(0, 1)

        decision = "Fail to reject"
        if accuracy_of_prediction <= 0.95:
            decision = "Reject"

        return decision, _max[0]
        

    def fit(self, x, y):
        self.x = x
        self.y = y

        #compute the list of all words in each class
        # freq_dict == {class: {word:freq}, class: {...}, ...}
        # Contains set of all unique words and their relative frequency for each class
        freq_dict = {}
        for domain in set(y):
            freq_dict[domain] = []

        for i, domain in enumerate(y):
            for word in x[i]:
                freq_dict[domain].append(word)

        #contains probability of all words given the class
        #-->{class: p(words | class}
        classifier = {}
        self.domain_means = {}
        self.print_dict = {}

        #take the set of each list
        for domain in freq_dict:
            print("\nComputing statistics on Domain:",domain)
            all_words = freq_dict[domain]
            set_all_words = list(set(all_words))
    
            #correlate the set to their relative frequencies
            corr_dict = {}
            for word in set_all_words:
                corr_dict[word] = 0

            all_words = sorted(all_words)
            for word in all_words:
                corr_dict[word] += 1
        
            #replace ordered words with summary of unique words relative to each word's frequency
            freq_dict[domain] = sorted(corr_dict.items(), key=operator.itemgetter(1))[::-1]

            #print statistics
            mean = int(np.mean(list(corr_dict.values())))
            stddev = int(np.std(list(corr_dict.values()), axis=0))
            print('Mean:',np.mean(list(corr_dict.values()),axis=0),
                  'Stdev:',np.std(list(corr_dict.values()), axis=0),
                  'Count:',len(set_all_words))
    
            #print('Words that occur most frequently:',[(pair[0], pair[1]) for pair in freq_dict[domain] if pair[1] == mean])
            self.domain_means[domain] = mean
            self.print_dict[domain] = freq_dict[domain]

            #keep words within certain range of the mean word freq
            spread = stddev #~98% of all words
            classifier[domain] = []
            for pair in freq_dict[domain]:
                if pair[1] in range(mean - spread, mean + spread):
                    classifier[domain].append([pair[0], pair[1]])

        [print('Percentage of target words kept:',domain, len(classifier[domain]),
                len(classifier[domain])*100//len(freq_dict[domain]),
                    '%')for domain in classifier]


        #determine inverse document frequency weighting
        doc_freq = {}
        for domain in classifier:
            for word, freq in classifier[domain]:
                if word not in doc_freq:
                    doc_freq[word] = 1
                else:
                    doc_freq[word] += 1
        self.doc_freq =sorted(doc_freq.items(), key=operator.itemgetter(1))[::-1]
        #convert classifier into probabilities of each word given the domain
        #  word_freq = (domain_freq * (num_domains / domain_freq))
        #  word_prob = sigmoid(word_freq / num_of_domains)
        for domain in classifier:
            total = 0
            #add the freq of each word to the total
            for pair in classifier[domain]:
                total += pair[1]

            total = np.sqrt(total)
            #divide frequencies through the total to get probabilies
            # --> P( class | word )
            for i, pair in enumerate(classifier[domain]):
                word_freq = np.sqrt(pair[1]) * (len(classifier) / doc_freq[word])
                classifier[domain][i] = [pair[0], self.sigmoid(word_freq/total)]

        self.freq_dict = {}
        #freq_dict = {domain:[ [word, P(word | domain)], ...]}
        for domain in classifier:
            #convert to hash lookup (dict) to optimize naive bayes
            self.freq_dict[domain] = dict(classifier[domain])

        

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%v#%#


cat_encoding = {'Break / Fix (Incident)':0,
                #'Information Request':1,
                #'Password Reset':2,
                'Catalog Order':3}

cat_decoding = {0:'Break / Fix (Incident)',
                #1:'Information Request',
                #2:'Password Reset',
                3:'Catalog Order'}
train_y = []
train_x = []
num_missed = 0
for inc in all_incidents:
    try:
        cat_encoding[inc.category]
        train_y.append(cat_encoding[inc.category])

        all_words = inc.sho_des + inc.description + inc.close_notes
        train_x.append(all_words)
    except KeyError as e:
        num_missed += 1
print('Percent of category not in cat_encoding', num_missed*100/len(train_x))

print('\nExecuting Missing Category Graph ...')

bayes = Bayesian_Classifier()

start_time = time.time()
bayes.fit(train_x, train_y)

cat_input_data = []
for inc in testing_incs:
    cat_input_data.append([inc.lexicon])
         
elapsed_time = time.time() - start_time

print('Time elapsed:', elapsed_time)

#bayes.evaluate_model()
bayes.test_accuracy(train_x, train_y)

cat_predictions = []
for inc in testing_incs:
    all_words = inc.sho_des + inc.description + inc.close_notes
    _, pred = bayes.classify(all_words)
    cat_predictions.append(pred)

cat_predictions = [cat_decoding[pred] for pred in cat_predictions]

########## END OF MISSING CATEGORY GRAPH ##########
                
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%v#%#
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%v#%#
                
#These will become one-hot encodings
sd_decode = {}
sd_encode = {#'Loaner':0, #<0%
             'Printing':1, #~21%
             'Peripherals':2, #~12%
             #'Mobile Device':3, #<1%
             #'Departmental System':4, #~1%
             #'Workstation Security':5, #~1%
             'Software Application':6, #~16%
             'Workstation Hardware':7, #~35%
             #'Classroom / Event Support':8, #~2%
             #'System / Application Access':9, #~5%
             #'Workstation OS Configuration':10, #~5%
             'Storage Permissions Configuration':11,#~6%
             }#'Workstation Network Configuration':12}#~3%

for sd in sd_encode:
    sd_decode[sd_encode[sd]] = sd

sd_train_x = []
sd_train_y = []
num_missed = 0
for inc in all_incidents:#[:3000]
    all_words = inc.sho_des + inc.description + inc.close_notes
    try:
        sd_train_y.append(sd_encode[inc.service_detail])    
        sd_train_x.append(all_words)

    except KeyError as e:
        num_missed += 1

print('\nExecuting Service Detail Graph ...')
print("Percent of training data excluded in this session:",num_missed*100/len(sd_train_x))
bayes = Bayesian_Classifier()

start_time = time.time()
bayes.fit(sd_train_x, sd_train_y)

sd_input_data = []
for inc in testing_incs:
    sd_input_data.append([inc.lexicon])

elapsed_time = time.time() - start_time

print('Time elapsed:', elapsed_time)

bayes.test_accuracy(sd_train_x, sd_train_y)

num_decisions = len(train_x)
num_incorrect_decisions = 0
for i, ins in enumerate(train_x[:num_decisions]):
    #print(bayes.classify(ins), train_y[i])
    dec, pred = bayes.classify(ins)

    #H0: correct classification
    if dec == 'Fail to reject':#H0 is true
        if pred != train_y[i]:
            num_incorrect_decisions += 1
            
    if dec == 'Reject':#h0 is false
        if pred == train_y[i]:
            num_incorrect_decisions += 1
            
print('Percentage of incorrect decisions:', 100*num_incorrect_decisions/num_decisions)



sd_predictions = []
for inc in testing_incs:
    all_words = inc.sho_des + inc.description + inc.close_notes
    dec, pred = bayes.classify(all_words)

    if dec == "Fail to reject":
        sd_predictions.append(sd_decode[pred])
    else:
        sd_predictions.append("verify: " + sd_decode[pred])

    
########## END OF SERVICE DETAIL GRAPH ##########

# Get reasons for update and then apply updates
# Populate output dictionaries (DataFrames)
for i, inc in enumerate(testing_incs):
    reasons = ""
    if inc.business_service == "Null":
        reasons += "Missing Business Service, "

    if inc.service_detail == "Null":
        reasons += "Missing Service Detail, "
        inc.service_detail = sd_predictions[i]
        
    if inc.category == "Null":
        reasons += "Missing Category, "
        inc.category = cat_predictions[i]

    inc.reasons = reasons

'''
########## Predictions of Resolution Categories ###########
########### END OF MACHINE LEARNING MODEL ###########
'''
res_cat_update = {'number':[],
                  'resolution_category':[]}

other_fields_update = {'Number':[],
                       'Category':[],
                       'CloseCode':[],
                       'ServiceDetail':[],
                       'ServiceProvider':['ITS' for inc in testing_incs],
                       'ServiceCatalogEntry':[]}

archive_agent_data = {'Number':[],
                      'Category':[],
                      'Close Code':[],
                      'Assigned To':[],
                      'Service Detail':[],
                      'Business Service':[],
                      'Service Provider':[],
                      'Assignment Group':[],
                      'Reason for Update':[],
                      'Resolution Category':[],
                      'Service Catalog Entry':[]}

###### compute statistics on output and validate ######
'''
for number in missing_res_cats:
    res_cat_update['number'].append(number)
    res_cat_update['resolution_category'].append(missing_res_cats[number])
   ''' 
for inc in testing_incs:
    archive_agent_data['Number'].append(inc.number)
    archive_agent_data['Category'].append(inc.category)
    archive_agent_data['Close Code'].append(inc.close_code)
    archive_agent_data['Assigned To'].append(inc.assigned_to)
    archive_agent_data['Service Detail'].append(inc.service_detail)
    archive_agent_data['Business Service'].append(inc.business_service)
    archive_agent_data['Service Provider'] = 'ITS'
    archive_agent_data['Assignment Group'].append(inc.assignment_group)

    #include resolved by data
    '''
    if inc.number in missing_res_cats:
        inc.reasons += 'Missing Resolution Category'
        inc.res_cat = missing_res_cats[inc.number]
    archive_agent_data['Resolution Category'].append(inc.res_cat)
    '''
    if inc.reasons == "":
        inc.reasons = "Not Updated"
    archive_agent_data['Reason for Update'].append(inc.reasons)
    
    archive_agent_data['Service Catalog Entry'].append(inc.serv_cat_ent)

def pred_serv_cat_ent(inc):
    if inc.serv_cat_ent == "":
        return inc.serv_cat_ent
    if inc.category != 'Catalog Order':
        return ""
    else:
        if inc.service_detail == 'Peripherals':
            return 'MiWorkspace Peripheral'
        elif inc.service_detail == 'Software Application':
            return 'MiWorkspace Software Installation'
        else:
            return 'Other'
        
for inc in testing_incs:
    if inc.reasons != "Not Updated":
        other_fields_update['Number'].append(inc.number)
        other_fields_update['Category'].append(inc.category)
        other_fields_update['CloseCode'].append(inc.close_code)
        other_fields_update['ServiceDetail'].append(inc.service_detail)
        other_fields_update['ServiceProvider'] = 'ITS'
        other_fields_update['ServiceCatalogEntry'].append(pred_serv_cat_ent(inc))
        


#res_cat_update = pd.DataFrame(res_cat_update)
other_fields_update = pd.DataFrame(other_fields_update)
#archive_agent_data = pd.DataFrame(archive_agent_data)

#res_cat_update.to_excel('./Output/res_cat_update_nov_10_test.xlsx') 
other_fields_update.to_excel('./Output/inc_cat_update_nov_20.xlsx')
#archive_agent_data.to_excel('./Output/archive_agent_data_nov_10_test.xlsx') 

print('\nProgram Completed Successfully.')
end = time.time() - initial_time
print('Total Runtime:', end)

import winsound
duration = 800  # millisecond
freq = 300  # Hz
winsound.Beep(freq, duration)




































