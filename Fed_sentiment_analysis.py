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

#link3 = r'C:/Users/alexn/Documents/Python_Projects/FOMC Sentiment Analysis/FOMC20100127meeting.pdf'
#text = (get_text(link3))

#len(text)
# print(text)

def get_transcript_links(base_url,base_directory,link_to_file_on_website=True,path_to_local_pdf=False,
                         path_to_local_txt=False,
                         years = range(1980,2024),transcript = True,
                         ):
    transcript_links = {}
    for year in range(1980, 2024): 
        
        if link_to_file_on_website:
            path = "fomchistorical" + str(year) + ".htm"
            html_doc = requests.get(base_url + path)
            soup = BeautifulSoup(html_doc.content, 'html.parser')
            if transcript:
                links = soup.find_all("a", string=re.compile('Transcript .*'))
            else:
                links = soup.find_all("a", href=re.compile('fomcminutes'),string=re.compile("PDF"))
                if year <2008:
                    links = soup.find_all("a", href=re.compile('minutes'))  
                    links = [x for x in links if "#" not in x["href"]]
                    if len(links)==0:
                        links = soup.find_all("a", href=re.compile('fomcmoa'))
                        if (year >=1993) & (year <=1996):
                            links = soup.find_all("a", href=re.compile('min')) 
                        
            link_base_url = "https://www.federalreserve.gov"
            if (year > 2018) & (transcript is False):
                path = 'fomccalendars.htm'
                html_doc = requests.get(base_url + path)
                soup = BeautifulSoup(html_doc.content, 'html.parser')
                links = soup.find_all("a", href=re.compile('fomcminutes.*'+str(year)),string=re.compile("PDF"))
            #link_base_url = "federalreserve.gov"
            transcript_links[str(year)] = [link_base_url + link["href"] for link in links]
            
        elif path_to_local_pdf or path_to_local_txt:
            files = []
            path_to_folder = base_directory  + str(year)
            new_files = os.walk(path_to_folder)
            for file in new_files:
                for f in file[2]:
                    if path_to_local_pdf:
                        if (f[-11:] == "meeting.pdf") & (transcript):
                          files.append(str(file[0]) + "/" + f)
                        elif ("minutes" in f) & (transcript is False):
                          files.append(str(file[0]) + "/" + f)
                          
            transcript_links[str(year)] = files
        print("Year Complete: ", year)
    return transcript_links


"""### Grab all the files and extract the text from the pdf"""

def grabFiles():
    #Pull the files
    
    # generates a dictionary of transcript paths
    # if we already have the pdf data, set path_to_local_pdf to True. 
    link_to_file_on_website = False
    path_to_local_pdf = True
    path_to_local_txt = False
    
    base_directory = 'C:/Users/alexn/Documents/Python_Projects/FOMC Sentiment Analysis/'
    # =============================================================================
    # ##resetting to grab from web ANS
    # link_to_file_on_website = True
    # path_to_local_pdf = False
    # path_to_local_txt = False
    # =============================================================================
    
    base_url = "https://www.federalreserve.gov/monetarypolicy/"
    base_directory = 'C:/Users/alexn/Documents/Python_Projects/FOMC Sentiment Analysis/'
 


    
    link_to_file_on_website = True
    path_to_local_pdf = False   
    transcript = False
    transcript_links = get_transcript_links(base_url,base_directory,link_to_file_on_website,path_to_local_pdf,
                                            path_to_local_txt,
                                            transcript=transcript)
    
    if link_to_file_on_website:
        for year in transcript_links:
            try:
                os.chdir(r"C:/Users/alexn/Documents/Python_Projects/FOMC Sentiment Analysis/"+str(year))
            except:
                os.mkdir(r"C:/Users/alexn/Documents/Python_Projects/FOMC Sentiment Analysis/"+str(year))   
                os.chdir(r"C:/Users/alexn/Documents/Python_Projects/FOMC Sentiment Analysis/"+str(year))
            for link in transcript_links[year]:
                if os.path.isfile(link.split('/')[-1]):
                    print("File ", link, " exists")
                else:
                    response = requests.get(link)
                # Write content in pdf file
                    if (transcript is False) & ("fomcminutes" not in link.split('/')[-1]):
                        pdf = open("fomcminutes"+link.split('/')[-1], 'wb')
                    else:
                        pdf = open(link.split('/')[-1], 'wb')
                    pdf.write(response.content)
                    pdf.close()
                    print("File ", link, " downloaded")
    
    Dates, transcripts, all_words, top_tokens = [],[],[],[]
    
    transcript_links = get_transcript_links(False,True,transcript=transcript)

    #transcript_links contains every pdf in each year
    for year in transcript_links: #produces the year index folder 
    
      #if int(year)<2010:continue
      # if int(year) == 2010: break #test out one decade - opted to end study at 2010
    
      for file in transcript_links[year]:  #the file in each year folder
        print(file)
        if ('minority' in file) or ('#' in file):
            continue
        text = get_text(file)
        words = get_words(text)
        
        #Append datapoints to respective lists
       # Dates.append(datetime.strptime(file[-19:-11], '%Y%m%d').date())
        Dates.append(dparser.parse(file.split("/")[-1],fuzzy=True))
        transcripts.append(text)  #all pages?
        all_words.append(words[0])  #returns all words in the document
        top_tokens.append(words[1]) #tuple of words with their respective counts in that document
     
    #Formatting data in to dataframe
    Data = pd.DataFrame([Dates,transcripts, all_words, top_tokens]).T
    Data.columns =['Date','Transcript','all_words', 'top_tokens']
    #Data.set_index('Date', inplace= True)
    #Data.sort_index(inplace=True)
    Data = Data.loc[:,:'Transcript'] ##file ends up being too large to git

    #save our dataset down for later analysis
    
    #Data.to_csv('/content/drive/MyDrive/Colab Notebooks/Projects/FOMC_NLP/fomc_transcript_tokens', header=True)
    if transcript:
        fileName = 'C:/Users/alexn/Documents/Python_Projects/FOMC Sentiment Analysis/fomc_transcript_tokens'
    else:
        fileName = 'C:/Users/alexn/Documents/Python_Projects/FOMC Sentiment Analysis/fomc_minutes_tokens'
    
    Data.to_csv(fileName, header=True)

"""To preserve the exact structure of the DataFrame, an easy solution 
is to serialize the DF in pickle format with pd.to_pickle, instead 
of using csv, which will always throw away all information about data types, 
and will require manual reconstruction after re-import. One drawback of pickle is 
that it's not human-readable.
repr() and ast.literal_eval(); for just lists, tuples and integers since 
the csv format converts everything to string.
"""


from ast import literal_eval  

#reload our transcript tokens csv
Data = pd.read_csv(fileName)

# Data['Transcript'] = Data['Transcript'].apply(literal_eval)
Data['all_words'] = Data['all_words'].apply(literal_eval)
Data['top_tokens'] = Data['top_tokens'].apply(literal_eval)
Data['Date'] = pd.to_datetime(Data['Date'], format='%Y-%m-%d')
Data.set_index("Date", inplace = True)

#Count the total word frequency across all transcripts
word_df = pd.DataFrame(columns=['Words', 'Count'])
word_df = []
for i in range(len(Data)):  #total links
  word_count = {'Words':[],'Count':[]}
  for sets in Data['top_tokens'][i]:
    # print(files)  #total pairs
    word_count['Words'].append(sets[0])  #total words broken out
    word_count['Count'].append(sets[1])  #total count per word
    
  word_df1 = pd.DataFrame(word_count)
  word_df.append(word_df1)

word_df = pd.concat(word_df)

total_words = word_df.groupby(['Words']).sum()
sorted_top_words = total_words.sort_values(by='Count', ascending=False)
sorted_top_words=sorted_top_words[:25]

sorted_top_words

import matplotlib.pyplot as plt

sorted_top_words = sorted_top_words.sort_values(by="Count")

plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',axisbelow=True, grid=True)
plt.rc('grid', color='w', linestyle='solid')

ax = sorted_top_words.plot(kind='barh', figsize=(8, 10), color='#86bf91', width=0.85);

ax.get_legend().remove()
plt.title("FOMC Minutes - Frequently used words from 1980 to 2010",fontsize=12, weight='bold')
# Draw vertical axis lines
vals = ax.get_xticks()
for tick in vals:
    ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

# Set axis labels
ax.yaxis.label.set_visible(False)
ax.set_xlabel("Total number of occurences", labelpad=20, weight='bold', size=10)
plt.savefig('fomc_top_words.png',dpi=60, bbox_inches = "tight")
#files.download('fomc_top_words.png');

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
##not used
# =============================================================================
# #Create a matrix of word types and the words that match these types
# word_list = []
# for sentiment_class in ["Negative", "Positive", "Uncertainty",
#                        "StrongModal", "WeakModal", "Constraining"]:
#     sentiment_list = pd.read_excel("/content/drive/MyDrive/Colab Notebooks/Projects/FOMC_NLP/LM Word List.xlsx", sheet_name=sentiment_class,header=None)
#     sentiment_list.columns = [sentiment_class]
#     sentiment_list[sentiment_class] = sentiment_list[sentiment_class].str.lower()
#     word_list.append(sentiment_list)
# word_list = pd.concat(word_list, axis=1, sort=True).fillna(" ")
# print(word_list.head())
# word_list = word_list.to_dict('list')  #create a dictionary out of the excel list and use it to map the transcripts
# print(word_list)
# 
# =============================================================================
import sys
sys.path.append(r'C:/Users/alexn/Documents/Python_Projects/FOMC Sentiment Analysis/')
import Loughran_MacDonald as LM
md = r"C:\Users\alexn\Documents\Python_Projects\FOMC Sentiment Analysis\Loughran-McDonald_MasterDictionary_1993-2021.csv"

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

Data['wordcount'] = temp.iloc[:,0].values
Data['NPositiveWords'] = temp.iloc[:,1].values
Data['NNegativeWords'] = temp.iloc[:,2].values
Data['Poswords'] = temp.iloc[:,3].values
Data['Negwords'] = temp.iloc[:,4].values

temp.head()

plt.rcParams["figure.figsize"] = (18,9)
plt.style.use('fivethirtyeight')

fig, ax = plt.subplots()

ax.plot(Data.index, Data['NPositiveWords'], 
         c = 'green',
         linewidth= 1.0)

plt.plot(Data.index, Data['NNegativeWords'], 
         c = 'red',
         linewidth=1.0)

plt.title('Highly correlated use of both good and bad words')

plt.legend(['Count of Positive Words', 'Count of Negative Words'],
           prop={'size': 20},
           loc = 2
           )

# format the ticks
# round to nearest years.
import matplotlib.dates as mdates
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

datemin = np.datetime64(Data.index[0], 'Y')
datemax = np.datetime64(Data.index[-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.grid(True)

plt.savefig('fomc_correlated_words.png',dpi=60, bbox_inches = "tight")
#files.download('fomc_correlated_words.png');

plt.show()

NetSentiment = (Data['NPositiveWords'] - Data['NNegativeWords'])

fig, ax = plt.subplots()

ax.plot(Data.index, NetSentiment, 
         c = 'red',
         linewidth= 1.0)

plt.title('Net sentiment implied by BoW over time',size = 'medium')

# format the ticks
# round to nearest years.
datemin = np.datetime64(Data.index[0], 'Y')
datemax = np.datetime64(Data.index[-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.grid(True)

plt.savefig('fomc_net_sentiment.png',dpi=60, bbox_inches = "tight")
#files.download('fomc_net_sentiment.png');

plt.show()

firstderivative = (NetSentiment.shift(1) / NetSentiment)

fig, ax = plt.subplots()

ax.plot(Data.index, firstderivative, 
         c = 'red')

plt.title('Change in sentiment over time (first derivative)')

# format the ticks
# round to nearest years.
datemin = np.datetime64(Data.index[0], 'Y')
datemax = np.datetime64(Data.index[-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.grid(True)

plt.savefig('fomc_sentiment_chg.png',dpi=60, bbox_inches = "tight")
#files.download('fomc_sentiment_chg.png');

plt.show()

#1979 Iranian Oil Crisis 
#https://en.wikipedia.org/wiki/1979_oil_crisis
Oil = np.logical_and(Data.index > '1980-01',
                          Data.index < '1982-11'
                          )

#Black Monday and the time period till US Equity market recovery
#https://en.wikipedia.org/wiki/Black_Monday_(1987)
BlkMonday = np.logical_and(Data.index > '1987-10',
                          Data.index < '1989-09'
                          )
#1994-1995 Mexican Peso crisis
#https://en.wikipedia.org/wiki/1998_Russian_financial_crisis
Peso = np.logical_and(Data.index > '1994-12',
                       Data.index < '1995-08'
                       )

#1998–1999 Russian Ruble crisis
#https://en.wikipedia.org/wiki/1998_Russian_financial_crisis
Russian = np.logical_and(Data.index > '1998-08',
                       Data.index < '1999-08'
                       )

#Dot-com bubble
#https://en.wikipedia.org/wiki/Dot-com_bubble
DotCom = np.logical_and(Data.index > '2000-03',
                         Data.index < '2002-10'
                        )

#Financial crisis of 2007–2008
#https://en.wikipedia.org/wiki/Financial_crisis_of_2007%E2%80%932008
Credit = np.logical_and(Data.index > '2007-04',
                         Data.index < '2009-03'
                        )


Crisis = np.logical_or.reduce( ( Oil,BlkMonday,Peso, Russian,DotCom,Credit) )

Window = 8
CompToMA = NetSentiment.rolling(Window).mean()

fig, ax = plt.subplots()
ax.plot(Data.index,
         CompToMA,
         c = 'r',
         linewidth= 2)

ax.plot(Data.index, NetSentiment, 
         c = 'green',
         linewidth= 1,
         alpha = 0.5)

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)
# round to nearest years.
datemin = np.datetime64(Data.index[0], 'Y')
datemax = np.datetime64(Data.index[-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.grid(True)


plt.title( str('Moving average of last ' + str(Window) + ' statements (~1 Year Window) coicides with periods of economic uncertainty / systemic risk:'))

ax.legend([str(str(Window) + ' statement MA'), 'Net sentiment of individual statements'],
           prop={'size': 20},
           loc = 2
          )

import matplotlib.transforms as mtransforms
trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
theta = 0.9
ax.fill_between(Data.index, 0, 10, where = Crisis,
                facecolor='grey', alpha=0.5, transform=trans)

xs = Data.index
ys = NetSentiment
ax.annotate('1979 Iranian \n Oil Crisis',
            (mdates.date2num(xs[2]), -800))
ax.annotate('Black \n Monday',
            (mdates.date2num(xs[66]), -600))
ax.annotate('Mexican \nPeso\nCrisis',
            (mdates.date2num(xs[122]), -800))
ax.annotate('Russian \nRuble',
            (mdates.date2num(xs[152]), -875))
ax.annotate('Dot-Com',
            (mdates.date2num(xs[165]), -900))
ax.annotate('US Financial \nCrisis',
            (mdates.date2num(xs[222]), -1250))


plt.savefig('fomc_crisis_periods.png',dpi=60, bbox_inches = "tight")
#files.download('fomc_crisis_periods.png');
plt.show()

#Retrive historical price data for the SP500
from datetime import datetime, timedelta
ticker = '^GSPC'

window = 250
start = Data.index[0] - timedelta(days=window)
end = Data.index[-1]
#market = yf.download(ticker, start=start, end=end, auto_adjust=False)



####next section sppeches analysis#########################
###grab meeting minutes

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

SentData['Net Sentiment'] = SentData['NPositiveWords']-SentData['NNegativeWords']
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

