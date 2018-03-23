# Identify specific Incidents as being incorrectly classified as service center
#     Identify change of error over time
# Developed by Nathan Shepherd

print('Importing dependencies ...')
import warnings # Prevents Gensim from printing an error on EVERY RUN
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=UserWarning, module='ipykernel_launcher')

from gensim import corpora, models, similarities
from nltk import word_tokenize, sent_tokenize
from stopwords import stopwords
import pandas as pd
import pickle

import codecs
import os

'''####################################'''
'''          Preprocess Data           '''

print('\nImporting Service Center data ...')

file = codecs.open('./All_Reports/service_center_incs.csv', "r",encoding='utf-8', errors='ignore')
serv_cent = pd.read_csv(file)
file.close();#print(serv_cent.head())

for field in serv_cent:
    serv_cent[field].fillna('Null', inplace=True)

testing_data = {'number':serv_cent['Number'].tolist(),
                'closed':serv_cent['Closed'].tolist(),
                'sho_des':serv_cent['Short description'].tolist(),
                'description':serv_cent['Description'].tolist(),
                'business_service':serv_cent['Business Service'].tolist(),
                'assignment_group':serv_cent['Assignment group'].tolist()}

#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#

length_test = len(serv_cent['Number'].tolist())#~29000
print('--> Rows in service_center_compiled:',length_test)

#%#%#%#%#%##%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#
# Store all words as Incident objects

class Incident():
    lexicon = [] #Array representing input (unique matricies in model)
    relativity = {} #relates this incident to overall ground truth
    def __init__(self, number, closed, sho_des, description, business_service, assignment_group):
        self.number = number
        self.closed = closed
        self.sho_des = sho_des
        self.description = description
        self.business_service = business_service
        self.assignment_group = assignment_group
        
        
        self.database = {'':[self.number],
                         '===Business Service':business_service,
                         '===Short Description':sho_des,
                         '===Description':description}
    def show(self):
        for field in self.database:
            print(field, self.database[field])

print('\nCreating Incident objects ...')

serv_cent_incs = []
for i in range(length_test):
    serv_cent_incs.append(Incident(testing_data['number'][i],
                                   testing_data['closed'][i],
                                   testing_data['sho_des'][i],
                                   testing_data['description'][i],
                                   testing_data['business_service'][i],
                                   testing_data['assignment_group'][i]))
##pickle.dump( serv_cent_incs, open( "service_center_incidents.pkl", "wb" ))
##print('\nSuccessfully exported service center objects\n')

biz_services = {'Google':['google', 'm+google'],
                'ServiceLink':['servicelink'],
                'Call Disconnected':['call', 'disconnected', 'lost', 'dropped'],
                'Customer Abandoned':['customer','abandoned'],
                'Desktop Support':['desktop','support','home'],
                'Printing':['printing','printer','print','scanner'],
                'Redirect or Transferred':['transferred','redirect','4help'],
                'Campus Computing Sites':['angel','computing','computer','windows'],
                'Exchange Email & Calendar':['email','e-mail','emails','mail','relay'],            
                'Identity and Access Management Application Services':['uniqname', 'access', 'account', 'accounts']}
'''
Medical / Hospital Exchange (UMHS Exchange)
    Nursing, Pharmacy, Dentistry and UHS connecting to UMHS Exchange via Outlook for Med Mail (OWA)
        Nursing, Pharmacy and UHS customers
             Business Service = MiWorkspace, Service Detail = Software Application
    Dentistry customers
        Business Service = Service Center 
        Best effort assistance, then redirect to Dentistry local IT
    Other customers --> Service Center
'''


freq_words = {}
classification = []
biz_service_exist = []
biz_class = []
def identify_others():
    print('Identifying Accurate Business Services')
    #update modified excel in access and accounts
    #in new column bool val if BS exists in servicelink
        #check against all BS that exists
        #check out service portfolio
    #add new column analysis for different
    for i, inc in enumerate(serv_cent_incs):
        if i % (len(serv_cent_incs)//5) == 0 and i > 0:
            print('--> {} %'.format(((i*100)//len(serv_cent_incs)) + 1))
        accurate_biz = {#if talking of medical related subject, service center
                        #    See if medical terms occur often
                        'Google':0,
                        'Printing':0,
                        'ServiceLink':0,
                        'Desktop Support':0,
                        'Call Disconnected':0,
                        'Customer Abandoned':0,
                        'Campus Computing Sites':0,
                        'Redirect or Transferred':0,
                        'Exchange Email & Calendar':0,
                            #filter all incidents containing:
                            #    "contact user by email" --> serv_cent
                            #        this text can be ignored
                            # UMHS, MSIS, MCIT, HITS --> Service Center == Biz service
                            #    biz_class --> med exchange (UMHS, MSIS, MCIT, HITS, med_syn)
                        'Identity and Access Management Application Services':0}
                        ####
                        ## Add Proposed Service Detail for all:
                        ##     serv_cent_incs.biz_serv_exists == 'No' <-- Proposed Biz Serv

        if "Contact user by Email" in inc.description:
            inc.description = inc.description[:-len("Contact user by Email")]
        all_words = word_tokenize(inc.sho_des +' '+ inc.description)
        for word in all_words:
            if word not in freq_words:
                freq_words[word] = 0
            else:
                freq_words[word] += 1
        appendage = []
        for word in all_words:
            for service in biz_services:
                for syn in biz_services[service]:
                    if service == 'Google':
                        if word.lower() == syn.lower():
                            accurate_biz[service] += 1
                    if service == 'Printing':
                        if word.lower() == syn.lower():
                            accurate_biz[service] += 1
                    if service == 'ServiceLink':
                        if word.lower() == syn.lower():
                            accurate_biz[service] += 1
                    if service == 'Campus Computing Sites':
                        if word.lower() == syn.lower():
                            accurate_biz[service] += 1
                    if service == 'Exchange Email & Calendar':
                        if word.lower() == syn.lower():
                            accurate_biz[service] += 1
                    if service == 'Customer Abandoned':
                        if word.lower() == syn.lower():
                            accurate_biz[service] += 1
                    if service == 'Identity and Access Management Application Services':
                        if word.lower() == syn.lower():
                            accurate_biz[service] += 1
                    if service == 'Call Disconnected':
                        if word.lower() == syn.lower():
                            accurate_biz[service] += 1
                    if service == 'Redirect or Transferred':
                        if word.lower() == syn.lower():
                            accurate_biz[service] += 1
                    if service == 'Desktop Support':
                        if word.lower() == syn.lower():
                            accurate_biz[service] += 1
        classifier = ''
        _max = 0
        for key in accurate_biz:
            if accurate_biz[key] > _max:
                _max = accurate_biz[key]
                classifier = key
                
        if classifier == 'Customer Abandoned':
            if 'Abandoned' not in all_words:
                classifier = 'Service Center'
                appendage = ['Other service not identified (Customer)']
                
        if accurate_biz['Customer Abandoned'] == 2:
            if 'Customer' in all_words and 'Abandoned' in all_words:
                classifier = 'Customer Abandoned'
                appendage = ['Customer Abandoned']


        if classifier == 'Call Disconnected':
            identified = 0
            found = False
            for word in all_words:
                if word.lower() in ['disconnected','lost','dropped']:
                    found = True
            if found:
                classifier = 'Call Disconnected'
                appendage = ['Contains two of: call, disconnected, lost, dropped']
            else:
                classifier = 'Service Center'
                appendage = ['Other service not identified (Call)']
                

        if classifier == 'Exchange Email & Calendar':
            for word in all_words:
                if word == 'm+google':
                    classifier = 'Google'
                    appendage == ['M+Google']
                   
        if classifier == 'Printing':
            if ('recycle' in all_words) or ('cartridge' in all_words) or ('home' in all_words) or ('retiree' in all_words) or ('alumni' in all_words):
                #if home printer and alumni or retiree -> service center
                #if printer configure and miworkspace customer BS -> not service center
                classifier = 'Service Center'
                appendage = ['Contains word like \'print\' and one of: recycle, cartridge, home, retiree, alumni']
            else:
                classifier = 'Printing, not Service Center'
                appendage = ['Contains word like \'print\'']

        if classifier == '' or _max == 0:
            classifier = 'Service Center'
            appendage = ['Other service not identified']
        
        if classifier == 'Requires Manual Classification':
            appendage = ['Other service not identified']
            classifier = 'Service Center'
            for word in all_words:
                if word in freq_words:
                    freq_words[word] += 1
                else:
                    freq_words[word] = 0

        if classifier not in ['Service Center','Printing, not Service Center','Call Disconnected','Desktop Support']:
            for syn in biz_services[classifier]:
                for word in all_words:
                    if word.lower() == syn.lower() and word.lower() not in appendage:
                        appendage.append(word.lower())
            appendage = sorted(list(set(appendage)))#set() removes duplicate words
            #### 
        out_str = ''
        for word in appendage:
            if len(word) > 1:
                out_str += word + ' '
            else:
                out_str += word
        ###
        if classifier == 'Redirect or Transferred' and len(appendage) == 1:
            if appendage[0] == '4help':
                classifier = 'Service Center'
                appendage = 'Other service not identified (4help)'
                out_str = 'Other service not identified (4help)'

        if inc.number == 'ad,':
            out_str = 'servicelink'

        if classifier in ['Call Disconnected','Customer Abandoned','Redirect or Transferred','Printing, not Service Center']:
            biz_service_exist.append('No')
        else:
            biz_service_exist.append('Yes')

        if classifier == 'Printing, not Service Center':
            biz_service_exist[-1] = 'No'
        if classifier == 'Campus Computing Sites':
            biz_service_exist[-1] = 'Yes'
        if classifier == 'Exchange Email & Calendar':
            biz_service_exist[-1] = 'Yes'
        if classifier == 'Google':
            biz_service_exist[-1] = 'Yes'
        if classifier == 'Service Center':
            biz_service_exist[-1] = 'Yes'
            
        biz_class.append(out_str)
        classification.append([inc.number, classifier])

identify_others()

all_strings = []
def filtration(arr, string):
    if string not in arr:
        arr.append(string)
##########
service_details = ['Unidentified' for i in range(len(serv_cent_incs))]
##########
exchange_dict = {}
for i, string in enumerate(classification):
    if string[1] == 'Service Center':
        if biz_class[i] == 'O t h e r   s e r v i c e   n o t   i d e n t i f i e d   ( 4 h e l p ) ':
            biz_class[i] = 'Other Service Not Identified (4help)'
            
#attempting to identify why three incidents show up with SD as business service
##        if biz_class[i] == 'Other service not identified (Customer) ':
##            biz_service_exist[i] = 'Yes'
##            if serv_cent_incs[i].sho_des == 'Customer Abandoned-Heard loud noises after she introduced herself, lost connect.':
##                classification[i][1] = 'Customer Abandoned'
##
##            if serv_cent_incs[i].sho_des == 'Call Disconnected':
##                classification[i][1] = 'Call Disconnected'

    if string[1] == 'Campus Computing Sites':
        if 'angel' not in biz_class[i]:
            classification[i][1] = 'Service Center'
            service_details[i] = 'General Computing'

    if string[1] == 'Desktop Support':
            all_words = word_tokenize(serv_cent_incs[i].sho_des +' '+ serv_cent_incs[i].description)
            classification[i][1] = 'Service Center'
            found = False
            for word in all_words:
                if word.lower() == 'home':
                    service_details[i] = 'Personal Device Support'
                    biz_class[i] = 'Home and possibly one of: desktop or support'
                    found = True
                
            for word in all_words:
                if word.lower() in ['med','medicine','medical','clinic','clinical','UMHS']:
                    service_details[i] = 'Non-ITS Desktop Support'
                    biz_class[i] = 'Contains medical synonymns'
                    found = True
            if not found:
                biz_class[i] = 'Other service not identified'
            
    if string[1] == 'Identity and Access Management Application Services':
        if 'User Account is Compromised' in serv_cent_incs[i].sho_des:
            service_details[i] = 'Compromised Account'
            classification[i][1] = 'Service Center'
    
    if string[1] == 'Exchange Email & Calendar':
        all_words = word_tokenize(serv_cent_incs[i].sho_des +' '+ serv_cent_incs[i].description)
        classified = False
        for word in all_words:
            if not classified:
                if word.lower() == 'voicemail':
                    classification[i][1] = 'Voice Services'
                    service_details[i] = 'Voicemail'
                    biz_class[i] = 'voicemail'
                    classified = True
                elif word.lower() in ['nursing', 'pharmacy', 'UHS']:
                    classification[i][1] = 'MiWorkspace'
                    service_details[i] = 'Software Application'
                    classified = True
                elif word.lower() == 'dentistry':
                    classification[i][1] = 'Service Center'
                    classified = True

        if not classified:
            #only identify - MAIL RELAY SERVICE REGISTRATION FORM SUBMITTED
            service_details[i] = 'Not UM Google Email'
            classification[i][1] = 'Service Center'
            biz_class[i] == 'Other service not identified (Not UM Google Email)'
            

    if string[1] == 'Customer Abandoned':
        classification[i][1] = 'Service Center'
        service_details[i] = 'Customer Abandoned'
        biz_class[i] = 'Customer Abandoned'

    if string[1] == 'Identity and Access Management Application Services':
        filtered = []
        words = word_tokenize(biz_class[i])
        for i, word in enumerate(words):
            if i < len(words) - 1:
                if not word[i] + 's' == word[i + 1] :
                    if word[i] not in filtered:
                        filtered.append(word)
        out_str = ''
        for word in filtered:
            out_str += word + ' '
        biz_class[i] = out_str

biz_class[0] = 'servicelink'
for i, inc in enumerate(serv_cent_incs):
    #and word_tokenize(biz_class[i])[-1].lower()[-2] == 'mail'
    if word_tokenize(biz_class[i])[-1].lower() == 'relay':
        service_details[i] = 'Mail Relay Service'

    for word in word_tokenize(inc.sho_des):
        if word.lower() == 'phish':
            service_details[i] = 'Phishing Attempt'
            #print('Found Phish')
            
    if service_details[i] == 'Not UM Google Email':
        for word in word_tokenize(inc.sho_des):
            if word == '4HELP':
                service_details[i] = 'Redirect Outside 4HELP'

            if word == 'MCIT':
                service_details[i] = 'Redirect Outside 4HELP'

            if word.lower() == 'mcommunity':
                service_details[i] = 'MCommunity'
                #print('Found MCommunity')

            if word.lower() == 'compromised':
                service_details[i] = 'Compromised Account'
    
            
    if inc.number == 'INC1008753':
        biz_service_exist[i] = 'No'
    if inc.number == 'INC1008753':
        biz_service_exist[i] = 'Yes'
    
    if inc.number == 'INC0000627':
        classification[i][1] = 'Identity and Access Management Application Services'
    if inc.number == 'INC0001010':
        classification[i][1] = 'Identity and Access Management Application Services'
    if inc.number == 'INC0001254':
        classification[i][1] = 'Identity and Access Management Application Services'

        
############
# Identify Service Details
service_details[2005] = 'Call Dropped'
service_details[6741] = 'Call Dropped'
service_details[17830] = 'Call Dropped'
for i, inc in enumerate(serv_cent_incs):
    if biz_service_exist[i] == 'No':
        if not classification[i][1] == 'Printing, not Service Center':
            service_details[i] = classification[i][1]
            classification[i][1] = 'Service Center'
        else:
            classification[i][1] = 'MiWorkspace'
            service_details[i] = 'Printing'

    if service_details[i] == 'Call Dropped':
        service_details[i] = 'Call Disconnected'
        
    if service_details[i] == 'Redirect or Transferred':
        service_details[i] = 'Redirect Outside 4HELP'

#No Change, Create Service Detail, Use Different Existing Business Service
proposed_changes = ['Null' for inc in range(len(serv_cent_incs))]
redirected_to = ['' for inc in range(len(serv_cent_incs))]#Proposed Redirect Destination
for i, inc in enumerate(serv_cent_incs):
    if service_details[i] == 'Redirect Outside 4HELP':
        proposed_changes[i] = 'Create Service Detail'

    ###################################
    if service_details[i] == 'Service Center':
        service_details[i] = 'Customer Abandoned'
        
    if classification[i][1] == 'Service Center':
        proposed_changes[i] = 'Create Service Detail'
        if service_details[i] == 'Unidentified':
            proposed_changes[i] = 'No Change'
    else:
        proposed_changes[i] = 'Use Different Existing Business Service'
    
            
##customer abandoned, service exists should be 'no'
## INC1008753
    

import operator
sorted_dict = sorted(freq_words.items(), key=operator.itemgetter(1))
##
##for pair in sorted_dict[-200:-100]:
##    print('\n',pair[0], pair[1],'\n')

out_data = pd.DataFrame({#'Binary Comparitor':[pair[0] for pair in classification],
                         'Biz Classification':[val for val in biz_class],
                         'Biz Service Exists?':['Yes' for val in biz_service_exist],
                         'Proposed Changes':[change for change in proposed_changes],
                         'Proposed Biz Service':[pair[1] for pair in classification],
                         #'Proposed Redirect Desitination':[val for val in redirected_to]
                         'Proposed Service Detail':[detail for detail in service_details]})
'''
'Serv_Cent Comparitor':[pair[0] for pair in conditions_used],
'Conditions used':[pair[1] for pair in conditions_used]
'''

serv_cent = pd.concat([serv_cent, out_data], axis=1)
#print(serv_cent.head())
serv_cent.to_excel('./working_directory/predicted_business_service_nov_10.xlsx')
















