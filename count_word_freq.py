from nltk import word_tokenize
import pandas as pd
import codecs
 
ci_canvas = pd.read_csv(codecs.open("CI_contains_Canvas.csv", "r",encoding='utf-8', errors='ignore'))
sd_ctools = pd.read_csv(codecs.open("service_detail_contains_ctools.csv", "r",encoding='utf-8', errors='ignore'))
sd_qualtrics = pd.read_csv(codecs.open("service_detail_Qualtrics.csv", "r",encoding='utf-8', errors='ignore'))

class Incident():
    words = []

    def __init__(self, number, short_des, description, category):

        self.number = number
        self.sho_des = short_des
        self.description = description
        self.category = category

def extract_incs(df, category):
    outs = []
    num_incs = len(df['number'])

    for i in range(num_incs):
        outs.append(Incident(df['number'][i],
                                    df['short_description'][i],
                                    df['description'][i],
                                    category))
    return outs

canvas_incs = extract_incs(ci_canvas, "CI contains Canvas")
ctools_incs = extract_incs(sd_ctools, "SD contains CTools")
qualtrics_incs = extract_incs(sd_qualtrics, "SD is Qualtrics")

for inc in canvas_incs:
    inc.words = word_tokenize(str(inc.description)+" "+str(inc.sho_des))
for inc in ctools_incs:
    inc.words = word_tokenize(str(inc.description)+" "+str(inc.sho_des))
for inc in qualtrics_incs:
    inc.words = word_tokenize(str(inc.description)+" "+str(inc.sho_des))

document_freq = {}
all_words = []
all_cats = {"canvas":[],
             "ctools":[],
             "qualtrics":[]}

for inc in canvas_incs:
    for word in inc.words:
        all_words.append(word)
        all_cats["canvas"].append(word)
for inc in ctools_incs:
    for word in inc.words:
        all_words.append(word)
        all_cats["ctools"].append(word)    
for inc in qualtrics_incs:
    for word in inc.words:
        all_words.append(word)
        all_cats["qualtrics"].append(word)

all_words = set(all_words)
all_cats = {"canvas":set(all_cats['canvas']),
            "ctools":set(all_cats['ctools']),
            "qualtrics":set(all_cats['qualtrics'])}
'''
timer = 0
for word in all_words:
    timer += 1
    if timer % 1000 == 0:
        print(timer / len(all_words))
    count = 0
    for check in all_cats['canvas']:
        if word == check:
            count += 1
            break
    for check in all_cats['ctools']:
        if word == check:
            count += 1
            break
    for check in all_cats['qualtrics']:
        if word == check:
            count += 1
            break
    document_freq[word] = count
'''
#Count term frequency
canvas_counts = {}; canvas_bigram = {}
ctools_counts = {}; ctools_bigram = {}
qualtrics_counts = {}; qualtrics_bigram = {}

stopwords = [':','.',',','>','-',')','(','--',"a",'n', "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the", "n"]

for inc in canvas_incs:
    for i in range(len(inc.words) - 1):
        bigram = inc.words[i]+" "+ inc.words[i + 1]
        if bigram in canvas_bigram:
            canvas_bigram[bigram] += 1
        else:
            canvas_bigram[bigram] = 1
            
    for word in inc.words:
        if word in canvas_counts and word not in stopwords:
            canvas_counts[word] += 1
        else:
            canvas_counts[word] = 1

for inc in ctools_incs:
    for i in range(len(inc.words) - 1):
        bigram = inc.words[i]+" "+ inc.words[i + 1]
        if bigram in ctools_bigram:
            ctools_bigram[bigram] += 1
        else:
            ctools_bigram[bigram] = 1
            
    for word in inc.words:
        if word in ctools_counts and word not in stopwords:
            ctools_counts[word] += 1
        else:
            ctools_counts[word] = 1
            
for inc in qualtrics_incs:
    for i in range(len(inc.words) - 1):
        bigram = inc.words[i]+" "+ inc.words[i + 1]
        if bigram in qualtrics_bigram:
            qualtrics_bigram[bigram] += 1
        else:
            qualtrics_bigram[bigram] = 1
            
    for word in inc.words:
        if word in qualtrics_counts and word not in stopwords:
            qualtrics_counts[word] += 1
        else:
            qualtrics_counts[word] = 2
'''
for word in canvas_counts:
    canvas_counts[word] = canvas_counts[word]//document_freq[word]

for word in ctools_counts:
    ctools_counts[word] = ctools_counts[word]//document_freq[word]

for word in qualtrics_counts:
    qualtrics_counts[word] = qualtrics_counts[word]//document_freq[word]
'''

import operator

canvas_counts = sorted(canvas_counts.items(), key=operator.itemgetter(1))[::-1]
ctools_counts = sorted(ctools_counts.items(), key=operator.itemgetter(1))[::-1]
qualtrics_counts = sorted(qualtrics_counts.items(), key=operator.itemgetter(1))[::-1]

canvas_bigram = sorted(canvas_bigram.items(), key=operator.itemgetter(1))[::-1]
ctools_bigram = sorted(ctools_bigram.items(), key=operator.itemgetter(1))[::-1]
qualtrics_bigram = sorted(qualtrics_bigram.items(), key=operator.itemgetter(1))[::-1]

canvas_df = pd.DataFrame({"CI contains Canvas":[str(word[0]) for word in canvas_counts],
                          "CI contains Canvas (counts)":[count[1] for count in canvas_counts]})
canvas_df = canvas_df.applymap(lambda x: x.encode('unicode_escape').
                 decode('utf-8') if isinstance(x, str) else x)
canvas_df.to_excel('canvas_counts.xlsx')

ctools_df = pd.DataFrame({"Serv. Det. contains CTools":[str(word[0]) for word in ctools_counts],
                          "Serv. Det. contains CTools (counts)":[count[1] for count in ctools_counts]})
ctools_df.to_excel('ctools_counts.xlsx')

qualtrics_df = pd.DataFrame({"Serv. Det. is Qualtrics":[str(word[0]) for word in qualtrics_counts],
                             "Serv. Det. is Qualtrics (counts)":[count[1] for count in qualtrics_counts],})
qualtrics_df.to_excel('qualtrics_counts.xlsx')





















