# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 22:15:26 2023

@author: alexn
"""



#from Github:
#https://github.com/gregjasonroberts/FOMC_NLP_Sentiment_Analysis/blob/main/src/fomc_nlp.py#L89

# =============================================================================
# !pip install BeautifulSoup4
# !pip install lxml
# !pip install html5lib
# !pip install nltk==3.2.5
# !pip install pdfplumber
# !pip install -U spacy
# !python -m spacy download en
# !pip install yfinance
# !pip install mpld3
# !pip install wordcloud
# =============================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import codecs
import dateutil.parser as dparser

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from collections import Counter
from nltk.corpus import stopwords
import spacy
from html import unescape
import pdfplumber
from datetime import datetime 
import datetime as dt

import urllib
import urllib.request
from bs4 import BeautifulSoup
from bs4 import re
import requests
import re
#import yfinance as yf
from statsmodels.tsa.stattools import adfuller

spacy.load('en_core_web_sm')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nlp = spacy.lang.en.English()
sp = spacy.load('en_core_web_sm')
all_stopwords = sp.Defaults.stop_words
all_stopwords |= {'the','is','th','s', 'm','would'}

# remove html entities from docs and set everything to lowercase
def my_preprocessor(doc):
  return(unescape(doc).lower())

# tokenize the doc and lemmatize its tokens
def my_tokenizer(doc):
    
    text = word_tokenize(doc)
    tokens_without_sw= [word for word in text if not word in all_stopwords]
    
    return tokens_without_sw

def preprocess_tokens(tokens):
  '''
  Remove any extra lines, non-letter characters, and blank quotes
  '''
  remove_new_lines = [re.sub('\s+', '', token) for token in tokens] 
  #Remove non letter characters
  non_letters = [re.sub('[^a-zA-Z]', '', remove_new_line) for remove_new_line in remove_new_lines]
  #Remove distracting single quotes
  remove_quotes = [re.sub("\'", '', non_letter) for non_letter in non_letters]
  #Removes empty strings from a list of strings
  final = list(filter(None, remove_quotes)) 
  
  return final

def extract_html(file):
   # with codecs.open(file, "rb", "utf-8") as f:
    with codecs.open(file, "rb") as f:
        html = f.read()
    
    soup = BeautifulSoup(html, "html.parser")
    
    for script in soup(["script", "style"]):
        script.extract()    # rip it out
    
    # get text
    text = soup.get_text()
    
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return text



def concat_text(text):
    x=''
    for i in text:
        x = x+i
    return x

def get_text(path):  #file is the path
  if path[-3:] == 'htm':
      return extract_html(path)
  docs = []
  with pdfplumber.open(path) as pdf:
    for i in range(len(pdf.pages)):
      page = pdf.pages[i]   
      text = page.extract_text()
      docs.append(text)
    concat = concat_text(docs)
  return concat
  
def get_words(full_text):

  raw = [word.lower() for word in full_text.split()]
 
  values = ','.join(map(str, raw))  #converts bytes object to string
  tokenizer = my_tokenizer(values)
 
  words = preprocess_tokens(tokenizer)
  # remove stopwords
  stops = nltk.corpus.stopwords.words('english')
  new_stopwords = ['chairman','would', 'mr']
  stops.extend(new_stopwords)
  words = [word for word in words if word not in stops]
  ##drop single letters
  words = [word for word in words if len(word)>1]
  counter = Counter()
  counter.update(words)
  most_common = counter.most_common(25) 

  return words, most_common


transcript = False
if transcript:
    fileName = 'fomc_transcript_tokens.csv'
else:
    fileName = 'fomc_minutes_tokens.csv'


from ast import literal_eval  

#reload our transcript tokens csv
Data = pd.read_csv(fileName)

for i in Data.index:
    all_words, top_tokens = get_words(Data.loc[i,'Transcript'])
    Data.loc[i,'all_words'] = all_words
    Data.loc[i,'top_tokens'] = top_tokens


# import matplotlib.pyplot as plt
# from matplotlib import animation
# import mpld3

# def barlist(n):
#     return [1 / float(n * k) for k in range(1, 6)]

# fig, ax = plt.subplots()
# n = 100  # Number of frames
# x = range(1, 6)
# barcollection = ax.bar(x, barlist(1))

# def animate(i):
#     y = barlist(i + 1)
#     for i, b in enumerate(barcollection):
#         b.set_height(y[i])

# anim = animation.FuncAnimation(fig, animate, repeat=False, blit=False, frames=n, interval=100)

# # Convert the animation to an HTML file
# html_output = mpld3.fig_to_html(fig)
# with open('barchart_animated.html', 'w') as html_file:
#     html_file.write(html_output)

# plt.show()




"""Build a dictionary:  Applying a sentiment analysis to the words in the documents using the Loughran-McDonald context-specific lexicon, which assigns a simple positive or negative value to words based on the financial services industry context"""

import Loughran_MacDonald as LM
md = r"Loughran-McDonald_MasterDictionary_1993-2021.csv"

master_dictionary, md_header, sentiment_categories, sentiment_dictionaries, stopwords, total_documents = \
    LM.load_masterdictionary(md, True, "", True)

word_list = sentiment_dictionaries

negate = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt", "ain't", "aren't", "can't",
          "couldn't", "daren't", "didn't", "doesn't", "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt",
          "neither", "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't", "neednt", "needn't",
          "never", "none", "nope", "nor", "not", "nothing", "nowhere", "oughtnt", "shant", "shouldnt", "wasnt",
          "werent", "oughtn't", "shan't", "shouldn't", "wasn't", "weren't", "without", "wont", "wouldnt", "won't",
          "wouldn't", "rarely", "seldom", "despite", "no", "nobody"]

def negated(word):
    """
    Determine if preceding word is a negation word
    """
    if word.lower() in negate:
        return True
    else:
        return False

def count_with_negation(fin_dict, transcript):
    """
    Count positive and negative words with negation check. Account for simple negation only for positive words.
    negation is occurring within three words preceding a positive words.
    """
    pos_count = 0
    neg_count = 0
 
    pos_words = []
    neg_words = []

    input_words = re.findall(r'\b([a-zA-Z]+n\'t|[a-zA-Z]+\'s|[a-zA-Z]+)\b', transcript.upper())
 
    word_count = len(input_words)
  

    for i in range(0, word_count):
      if input_words[i] in fin_dict['negative']:
       
        neg_count += 1
        neg_words.append(input_words[i])
      if input_words[i] in fin_dict['positive']:
        if i >= 3:
          if negated(input_words[i - 1]) or negated(input_words[i - 2]) or negated(input_words[i - 3]):
            neg_count += 1
            neg_words.append(input_words[i] + ' (with negation)')
          else:
            pos_count += 1
            pos_words.append(input_words[i])
        elif i == 2:
          if negated(input_words[i - 1]) or negated(input_words[i - 2]):
            neg_count += 1
            neg_words.append(input_words[i] + ' (with negation)')
          else:
              pos_count += 1
              pos_words.append(input_words[i])
        elif i == 1:
          if negated(input_words[i - 1]):
                neg_count += 1
                neg_words.append(input_words[i] + ' (with negation)')
          else:
                pos_count += 1
                pos_words.append(input_words[i])
        elif i == 0:
              pos_count += 1
              pos_words.append(input_words[i])
 
    results = [word_count, pos_count, neg_count, pos_words, neg_words]
 
    return results

temp = [count_with_negation(word_list,x) for x in Data.Transcript]
temp = pd.DataFrame(temp)

Data.set_index("Date",inplace=True)
Data['wordcount'] = temp.iloc[:,0].values
Data['NPositiveWords'] = temp.iloc[:,1].values
Data['NNegativeWords'] = temp.iloc[:,2].values
Data['Poswords'] = temp.iloc[:,3].values
Data['Negwords'] = temp.iloc[:,4].values
Data['NetSentiment'] = (Data['NPositiveWords'] - Data['NNegativeWords'])
Data['SentimentScore'] =(Data['NPositiveWords'] - Data['NNegativeWords']) / Data['wordcount']
Data['NetSentimentScore'] =(Data['NPositiveWords'] - Data['NNegativeWords']) / (Data['NPositiveWords'] + Data['NNegativeWords'])
Data['NetSentiment_chg'] = Data['NetSentiment'].shift(1) / Data['NetSentiment']

plt.rcParams["figure.figsize"] = (18,9)
plt.style.use('fivethirtyeight')


Data['NetSentiment'].plot()

Data['NetSentiment_chg'].plot()
plt.title('Change in sentiment over time (first derivative)')


Window = 8
CompToMA = Data['NetSentiment'].rolling(Window).mean()

fig, ax = plt.subplots()
Data['NetSentiment'].plot(ax=ax,c='green')
ax.plot(Data.index,
         CompToMA,
         c = 'r',
         linewidth= 2)



####next section sppeches analysis#########################
###grab meeting minutes
def grabSpeeches():
    from bs4 import BeautifulSoup
    import requests
    import pandas as pd
    #from textblob import TextBlob
    
    years=range(1996,2006)
    all_years = []
    for year in years:
        speeches_one_year = pd.DataFrame()
        page = requests.get(f'https://www.federalreserve.gov/newsevents/speech/{year}speech.htm')
        soup = BeautifulSoup(page.text, 'html.parser')
        title = soup.select(".title")
        speakers = soup.select(".speaker")
        locations = soup.select(".location")
        for i in range(len(title)):
            speeches_one_year.at[i,'link'] = 'https://www.federalreserve.gov'+title[i].find_all('a', href=True)[0]['href']
            speeches_one_year.at[i,'title'] = title[i].text.split('\n')[1]
            speeches_one_year.at[i,'speaker'] = speakers[i].text.split('\n')[1].strip()
            speeches_one_year.at[i,'event'] = locations[i].text.split('\n')[1].strip()
            speeches_one_year.at[i,'year'] = year
        all_years.append(speeches_one_year)
    
    
    years=range(2006,2025)
    for year in years:
        if year > 2010:
            page = requests.get(f'https://www.federalreserve.gov/newsevents/speech/{year}-speeches.htm')
        else:
            page = requests.get(f'https://www.federalreserve.gov/newsevents/speech/{year}speech.htm')
        soup = BeautifulSoup(page.text, 'html.parser')
        events = soup.select(".eventlist__event")
        speeches_one_year = pd.DataFrame()
        for i,speech in enumerate(events):
            speeches_one_year.at[i,'link'] = 'https://www.federalreserve.gov'+events[i].find_all('a', href=True)[0]['href']
            speeches_one_year.at[i,'title'] = events[i].text.split('\n')[2]
            if events[i].text.split('\n')[3]=='Watch Live' or events[i].text.split('\n')[3]=='Video':
                speeches_one_year.at[i,'speaker'] = events[i].text.split('\n')[4]
                speeches_one_year.at[i,'event'] = events[i].text.split('\n')[5]
            else:
                speeches_one_year.at[i,'speaker'] = events[i].text.split('\n')[3]
                speeches_one_year.at[i,'event'] = events[i].text.split('\n')[4]
                
            ##first split out
            tmp_txt = events[i].text.split('\n')
           # tmp_i = [i for i, s in enumerate(tmp_txt) if s.startswith('Governor') or 
          #           s.startswith('Chairman') or s.startswith('Vice Chairman')]
            #speeches_one_year.at[i,'speaker'] = tmp_txt[tmp_i[0]]
            tmp_txt = [x for x in tmp_txt if x not in ['','Watch Live','Video']]
            speeches_one_year.at[i,'event'] = tmp_txt[-1]#assume event last
            speeches_one_year.at[i,'title'] = tmp_txt[0]#assume title first
            speeches_one_year.at[i,'speaker'] = tmp_txt[1]
            speeches_one_year.at[i,'year'] = year
        all_years.append(speeches_one_year)
    
    all_years = pd.concat(all_years,axis=0,ignore_index=True)
    
    old_site_version_length = sum(all_years['year']<1999)
    for i in range(old_site_version_length):
        print(i)
        page = requests.get(all_years.loc[i,'link'])
        soup = BeautifulSoup(page.text, 'html.parser')
        text_list = [i for i in soup.find('p').getText().split('\n') if i] 
        text_list=text_list[:-8]
        text_list = ' '.join(text_list)
        text_list = text_list.replace('--', ' ')
        text_list = text_list.replace('\r', '')
        text_list = text_list.replace('\t', '')
        all_years.loc[i,'text'] = text_list
    
    for i in range(len(all_years)):
        if ((all_years.loc[i,'year']>1998) & (all_years.loc[i,'year']<2006)):
            print(i)
            page = requests.get(all_years['link'].iloc[i])
            soup = BeautifulSoup(page.text, 'html.parser')
            events = soup.select("table")
            if len(str(events[0].text))>600:
                text_list = [i for i in events[0].text if i] 
            else:
                text_list = [i for i in events[1].text if i]
            text_list = ''.join(text_list)
            text_list = text_list.replace('--', '')
            text_list = text_list.replace('\r', '')
            text_list = text_list.replace('\t', '')
            if ((i>=383) & (i<=536)):
                text_list = text_list.replace('     ', ' ')
                text_list = text_list.replace('    ', ' ')
            all_years.loc[i,'text'] = text_list
            
            
    black_listed = [742,748]
    for i in range(1,len(all_years)):
        if ((all_years.loc[i,'year']>2005) and (i not in black_listed)):
            print(i)
            page = requests.get(all_years.loc[i,'link'])
            soup = BeautifulSoup(page.text, 'html.parser')
            events = soup.select(".col-md-8")
            text_list = events[1].text
            text_list = text_list.replace('\xa0', ' ')
            text_list = text_list.replace('\n', ' ')
            all_years.loc[i,'text'] = text_list
    
    #all_years = pd.concat(all_years,axis=0,ignore_index=True)       
            
    all_years.loc[:,'date'] = all_years['link'].str.extract('(\d\d\d\d\d\d\d\d)')
    all_years = all_years[~all_years['text'].isna()]
    all_years.loc[:,'text_len'] = all_years['text'].str.split().apply(len)
    all_years = all_years[all_years['text_len']>5] ##drop speeches less than 5 characters
    all_years.loc[:,'location'] = all_years.event.str.split(', ').apply(lambda x: x[-1])
    
    dates = all_years['link'].str.extract('(\d\d\d\d\d\d\d\d)')
    all_years.loc[:,'date'] =pd.to_datetime(pd.Series(dates[0]))
    
    all_years.to_csv('speeches.csv')

loaded_speeches = pd.read_csv('speeches.csv')

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

text = loaded_speeches.iloc[-1]['text']
wordcloud = WordCloud().generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#wordcloud.to_file("img/test.png")

temp = [count_with_negation(word_list,x) for x in loaded_speeches['text']]
temp = pd.DataFrame(temp)

SentData = pd.DataFrame()

SentData['wordcount'] = temp.iloc[:,0].values
SentData['NPositiveWords'] = temp.iloc[:,1].values
SentData['NNegativeWords'] = temp.iloc[:,2].values * -1
SentData['Poswords'] = temp.iloc[:,3].values
SentData['Negwords'] = temp.iloc[:,4].values

SentData['NetSentiment'] = SentData['NPositiveWords']-SentData['NNegativeWords']
SentData['SentimentScore'] =(SentData['NPositiveWords'] - SentData['NNegativeWords']) / SentData['wordcount']
SentData['NetSentimentScore'] =(SentData['NPositiveWords'] - SentData['NNegativeWords']) / (SentData['NPositiveWords'] + SentData['NNegativeWords'])
SentData['NetSentiment_chg'] = SentData['NetSentiment'].shift(1) / SentData['NetSentiment']


SentData['date'] = loaded_speeches['date']
SentData.set_index('date',inplace=True)


##skikit-learn topic extraction
#https://blog.mlreview.com/topic-modeling-with-scikit-learn-e80d33668730

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % topic_idx)
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

documents = loaded_speeches['text']

no_features = 1000

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20
# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

no_top_words = 10
display_topics(nmf, tfidf_feature_names, no_top_words)
display_topics(lda, tf_feature_names, no_top_words)

##test change

