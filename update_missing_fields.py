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


print('\nConfiguring Incident Lexicons')

x_length = 25# 30?
def configure(in_incs):
    for i, inc in enumerate(in_incs):
        if i % 5000 == 0:
            print("--> ", (i*100)//len(in_incs),'%')
        
        all_words = []
        all_words = inc.sho_des + inc.description + inc.close_notes

        inc.lexicon = []
        for group in all_words:
            for word in group:
                inc.lexicon.append(model[word])

        random.shuffle(inc.lexicon)
        if len(inc.lexicon) > x_length:
            inc.lexicon = inc.lexicon[:x_length]
        
        if len(inc.lexicon) <= x_length:
            padding = []
            len_append = x_length - len(inc.lexicon)
        
            for i in range(len_append):
                index = random.randint(1, len(all_words) - 1)
                try:
                    padding.append(model[all_words[index]])
                except KeyError as e:
                    padding.append(model['a'])

            if len_append > 0:
                for word in padding:
                    inc.lexicon.append(word)
        inc.lexicon = np.array(inc.lexicon)
        inc.lexicon = np.reshape(inc.lexicon, -1) #1d vector:

configure(all_incidents)

print('Configuring Testing Incidents')
configure(testing_incs)

#%#%#%#%#%#% Neural Network #%#%#%#%#%#%
import tensorflow as tf
from math import floor

class NeuralComputer:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        print('init: len(x) = {}, len(y) = {}'.format(len(self.x), len(self.y)))
        
        self.in_dim = len(x[0])
        self.out_dim = len(y[0])
        
    def Perceptron(self, tensor):
        #with tf.name_scope('softmax_linear'):
            
        V0 = tf.Variable(tf.truncated_normal([self.in_dim, 1000]))
        b0 = tf.Variable(tf.truncated_normal([1000]))
        l0 = tf.sigmoid(tf.matmul(tensor, V0) + b0)

        V1 = tf.Variable(tf.truncated_normal([1000, 750]))
        b1 = tf.Variable(tf.truncated_normal([750]))
        l1 = tf.sigmoid(tf.matmul(l0, V1) + b1)

        V2 = tf.Variable(tf.truncated_normal([750, 600]))
        b2 = tf.Variable(tf.truncated_normal([600]))
        l2 = tf.sigmoid(tf.matmul(l1, V2) + b2)

        V3 = tf.Variable(tf.truncated_normal([600, 500]))
        b3 = tf.Variable(tf.truncated_normal([500]))
        l3 = tf.sigmoid(tf.matmul(l2, V3) + b3)

        V4 = tf.Variable(tf.truncated_normal([500, 300]))
        b4 = tf.Variable(tf.truncated_normal([300]))
        l4 = tf.sigmoid(tf.matmul(l3, V4) + b4)
        
        V5 = tf.Variable(tf.truncated_normal([300, 100]))
        b5 = tf.Variable(tf.truncated_normal([100]))
        l5 = tf.sigmoid(tf.matmul(l4, V5) + b5)

        V6 = tf.Variable(tf.truncated_normal([100, 25]))
        b6 = tf.Variable(tf.truncated_normal([25]))
        l6 = tf.sigmoid(tf.matmul(l5, V6) + b6)
        
        weights = tf.Variable( tf.zeros([25, self.out_dim]),name='weights')
        biases = tf.Variable(tf.zeros([self.out_dim]),name='biases')

        logits = tf.nn.softmax(tf.matmul(l6, weights) + biases)
        
        return logits, weights, biases

    def init_placeholders(self, n_classes, batch_size):
        #init Tensors: fed into the model during training
        x = tf.placeholder(tf.float32, shape=(None, self.in_dim))
        y_ = tf.placeholder(tf.float32, shape=(batch_size, n_classes))

        #Neural Network Model
        y, W, b = self.Perceptron(x)

        return y, W, b, x, y_

    def train(self, test_x, in_str, batch_size=1000, training_epochs=10,learning_rate=.5,display_step=1):
        print('train: len(x) = {}, len(y) = {}'.format(len(self.x), len(self.y)))
        print('len(test_x):',len(test_x))
        #batch_size = len(test_x)
        test_size = batch_size* floor(len(self.x)/batch_size)

        #to verify accuracy on novel data
        acc_x = self.x[test_size - batch_size*2:]
        acc_y = self.y[test_size - batch_size*2:]
        print("acc_x:",len(acc_x), ' acc_y:',len(acc_y))
        
        print("len_train",int(test_size - batch_size*2))
        self.x = self.x[:test_size - batch_size*2]
        self.y = self.y[:test_size - batch_size*2]
        
        # Train W, b such that they are good predictors of y
        self.out_y, W, b, self.in_x, y_ = self.init_placeholders(self.out_dim, batch_size)

        # Cost function: Mean squared error
        cost = tf.reduce_sum(tf.pow(y_ - self.out_y, 2))/(batch_size)

        # Gradient descent: minimize cost via Adam Delta Optimizer (SGD)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate,rho=.99,epsilon=3e-08).minimize(cost)

        # Initialize variables and tensorflow session
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(init)

        start_time = time.time()
        print_time = True
        for i in range(training_epochs):
            j=0
            while j < len(self.x):
                start = j
                end = j + batch_size
                
                self.sess.run([optimizer, cost], feed_dict={self.in_x: self.x[start:end],
                                                            y_: self.y[start:end]})
                j += batch_size
            # Display logs for epoch in display_step
            if (i) % display_step == 0:
                if print_time:
                    print_time = False
                    elapsed_time = time.time() - start_time
                    print('Predicted duration of this session:',(elapsed_time*training_epochs//60) + 1,'minute(s)')
                cc = self.sess.run(cost, feed_dict={self.in_x: acc_x[:batch_size], y_:acc_y[:batch_size]})
                print("Training step: {} || cost= {}".format(i,cc))
                        
        print("\nOptimization Finished!\n")
        training_cost = self.sess.run(cost, feed_dict={self.in_x: acc_x[:batch_size], y_:acc_y[:batch_size]})
        print("Training cost=",training_cost,"\nW=", self.sess.run(W)[:1],"\nb=",self.sess.run(b),'\n')
        correct_prediction = tf.equal(tf.argmax(self.out_y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuracy for predictions of {}'.format(in_str),
                self.sess.run(accuracy, feed_dict={self.in_x: acc_x[:batch_size], y_:acc_y[:batch_size]})*100,'%')
        
        #str(self.sess.run(accuracy, feed_dict={self.in_x: self.x[:batch_size], y_:self.y[:batch_size]})*100//1) + ' %'

    def save(self, in_str):
        self.saver.save(self.sess, in_str)

    def load(self, graph):
        #out_y, W, b, in_x, y_ = self.init_placeholders(self.out_dim, batch_size)
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph(graph + '.meta')
        self.saver.restore(self.sess, tf.train.latest_checkpoint('./'))
        
    def predict(self, test_x):
        predictions = []
        for matrix in test_x:
            predictions.append(self.sess.run(self.out_y, feed_dict={self.in_x:matrix}))

        self.sess.close()
        return predictions

    def max_of_predictions(self, predictions):
        out_arr = []
        for pred in predictions:
            #print('\n========')
            _max = [0, 0]# [index, value]
            for matrix in pred:
                for i, vect in enumerate(matrix):
                    if _max[1] <  vect:
                        _max[1] = vect
                        _max[0] = i
                    #print('{}::{}'.format(i,vect))
                #print(':MAX:', _max[0], _max[1])
            out_arr.append(_max[0])

        #indecies of max values in one-hot arrays
        return out_arr

#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%v#%#
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%v#%#


cat_encoding = {'Break / Fix (Incident)':[1, 0, 0, 0],
                'Information Request':[0, 1, 0, 0],
                'Password Reset':[0, 0, 1, 0],
                'Catalog Order':[0, 0, 0, 1]}

cat_decoding = {0:'Break / Fix (Incident)',
                1:'Information Request',
                2:'Password Reset',
                3:'Catalog Order'}
train_y = []
train_x = []
num_missed = 0
for inc in all_incidents:
    try:
        cat_encoding[inc.category]
        train_y.append(cat_encoding[inc.category])

        all_words = inc.sho_des + inc.description + inc.close_notes
        train_x.append(inc.lexicon)
    except KeyError as e:
        num_missed += 1
print('Num of category not in cat_encoding', num_missed)

print('\nExecuting Missing Category Graph ...')
nn = NeuralComputer(train_x, train_y)

start_time = time.time()

cat_input_data = []
for inc in testing_incs:
    cat_input_data.append([inc.lexicon])

nn.train(cat_input_data,"Missing Category",
         
         training_epochs=25, learning_rate=.05,
         
         batch_size=1000,display_step=3)
         
elapsed_time = time.time() - start_time

print('Time elapsed:', elapsed_time)

cat_predictions = nn.predict([[inc.lexicon] for inc in testing_incs])

cat_predictions = nn.max_of_predictions(cat_predictions)
cat_predictions = [cat_decoding[pred] for pred in cat_predictions]

########## END OF MISSING CATEGORY GRAPH ##########
                
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%v#%#
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%v#%#
                
#These will become one-hot encodings
sd_decode = {}
sd_encode = {'Loaner':0,
             'Printing':1,
             'Peripherals':2,
             'Mobile Device':3,
             'Departmental System':4,
             'Workstation Security':5,
             'Software Application':6,
             'Workstation Hardware':7,
             'Classroom / Event Support':8,
             'System / Application Access':9,
             'Workstation OS Configuration':10,
             'Storage Permissions Configuration':11,
             'Workstation Network Configuration':12}


for sd in sd_encode:
    decoder = sd_encode[sd]
    sd_decode[decoder] = sd

    label = np.zeros(13)#np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])
    label[sd_encode[sd]] = 1
    
    sd_encode[sd] = label


for inc in all_incidents:
    inc.sd_encoding = sd_encode[inc.service_detail]

sd_train_x = []
sd_train_y = []
for inc in all_incidents:#[:3000]
    sd_train_x.append(inc.lexicon)
    sd_train_y.append(inc.sd_encoding)

print('\nExecuting Service Detail Graph ...')
nn = NeuralComputer(sd_train_x, sd_train_y)

start_time = time.time()

sd_input_data = []
for inc in testing_incs:
    sd_input_data.append([inc.lexicon])

nn.train(sd_input_data,"Service Detail",
         
         training_epochs=35,learning_rate=.05,
         
         batch_size=1000, display_step=3)

elapsed_time = time.time() - start_time

print('Time elapsed:', elapsed_time)

sd_predictions = nn.predict([[inc.lexicon] for inc in testing_incs])

sd_predictions = nn.max_of_predictions(sd_predictions)
sd_predictions = [sd_decode[pred] for pred in sd_predictions]
    
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
other_fields_update.to_excel('./Output/inc_cat_update_nov_14.xlsx')
#archive_agent_data.to_excel('./Output/archive_agent_data_nov_10_test.xlsx') 

print('\nProgram Completed Successfully.')
end = time.time() - initial_time
print('Total Runtime:', end)

import winsound
duration = 1000  # millisecond
freq = 330  # Hz
winsound.Beep(freq, duration)




































