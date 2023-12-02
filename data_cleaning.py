# Importer les packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords') # Télécharger le package stopwords
nltk.download('wordnet')
from nltk.corpus import stopwords # Importer le package stopwords
import nltk
nltk.download('wordnet')

def remove_links(text):
    text=re.sub(r'http[s]*:?//\S+','',text)
    return text
def remove_at(text):
    
    text = re.sub(r'@[\w\-._]+','',text)
    return text
    
def special_carac(text):
    text = re.sub(r'[^a-zA-Z\s]+','',text)
    return text
def remove_RT(text):
  text= re.sub(r'RT','',text)
  return text

def upper_to_lower(text):
    text = re.sub(r'\s{2,}',' ',text)
    return text.lower()
# Fonction supprimant les stopwords
def stopword_removal(text):
    Stopwords = stopwords.words('english')
    text= ' '.join([word for word in text.split() if word not in Stopwords])
    return text
def cleaning_space(text):
    text = re.sub(r'^\s|\s$',' ' ,text)
    return text
from nltk.stem import WordNetLemmatizer
lemmatiser =WordNetLemmatizer()
def lemmat(text):
    text = [lemmatiser.lemmatize(word) for word in text.split()]
    text = " ".join(text)
    return text
def data_clean(df):
    
    df['new_text'] = df.text.apply(func =remove_links)
    df['new_text'] = df.new_text.apply(func =remove_at)
    df['new_text'] = df.new_text.apply(func =special_carac)
    df['new_text'] = df.new_text.apply(func =remove_RT)
    df['new_text'] = df.new_text.apply(func = upper_to_lower)
    df['new_text'] = df.new_text.apply(func =cleaning_space)
    df['new_text'] = df.new_text.apply(func =stopword_removal)
    df['new_text'] = df.new_text.apply(func =lemmat)
    return df

def text_cleaning(sentence):
    # Appliquer les transformations sur la phrase
    sentence = remove_links(sentence)
    sentence = remove_at(sentence)
    sentence = special_carac(sentence)
    sentence = remove_RT(sentence)
    sentence = upper_to_lower(sentence)
    sentence = cleaning_space(sentence)
    sentence = stopword_removal(sentence)
    sentence = lemmat(sentence)
    
    return sentence