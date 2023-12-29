
from logging import PlaceHolder
import streamlit as st 
from tensorflow.keras.models import load_model
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit  # 👈 Add the caching decorator
import nltk
nltk.download('stopwords')
import numpy as np
from data_cleaning import text_cleaning
import json
from keras.preprocessing.text import tokenizer_from_json
import langid
from langdetect import detect
st.title('Sentiment Analysis English vs French')
sentences = st.text_area("", placeholder="Entrer votre text")
#sentences=text_cleaning(sentences)
from util import config
import tensorflow as tf



# Définir la taille de la police souhaitée
font_size = "30px"

# Définir la police de caractères souhaitée
font_family = "Times New Roman, serif"


# Définir la couleur du texte souhaitée (code hexadécimal)
text_color = "white"

# Custom style settings
button_style = f"font-size: 50px; color: white; background-color: #2e6b8e; border-radius: 8px; padding: 10px 20px;"

result_text_style = f"font-size: 30px; color: white; font-family: 'Times New Roman', serif;"

config()
# Bouton de sélection de langue
language = st.radio("Choisissez la langue", ["English", "French"])

def loade_model(path):
        model=load_model(path)
        return model 
# Load the tokenizer from JSON
if st.button('Analyse sentiment',type='primary'):
    
  
    if language=="English":
        # ID du fichier sur Google Drive
        file_id ='1bS3R3rSyWLJswJI0M_qD1L1y-UGkNnjU'

        # URL de l'API Google Drive pour télécharger le fichier
        url = f'https://drive.google.com/uc?id={file_id}'

        # Utilisez la méthode `tf.keras.utils.get_file` pour charger le modèle directement depuis Google Drive
        model_path = tf.keras.utils.get_file('modelestop1.h5', origin=url, extract=False, cache_subdir='/content/drive/MyDrive/model/')

        # Chargez le modèle
        model = tf.keras.models.load_model(model_path,compile=False)
        
        with open('./tokenizer.json', 'r', encoding='utf-8') as json_file:
                
                tokenizer_json = json.load(json_file)
                tokenizer = Tokenizer()
                tokenizers = tokenizer_from_json(tokenizer_json)

        def my_reprocess(sentences, max_len=21):
                X =  tokenizers.texts_to_sequences(sentences)

                    # Pad the sequences
                X = pad_sequences(X, maxlen=max_len)

                return X
        
        # Preprocess the text
            
                #sentences = st.text_input("Enter text in French")
        processed_data = my_reprocess([sentences])

        #model = tf.keras.models.load_model('./model/modelestop1.h5',compile=False)
        y_pred = model.predict(processed_data)

        # Interpret the prediction
        probability = y_pred[0][0]
        if probability <= 0.5 :
            sentiment = "Positive sentiment"
        else:
            sentiment = "Negative"
        
        st.write(
                f'<div style="font-size: {font_size}; color: {text_color}; font-family: {font_family};">'
                f"Sentiment: {sentiment} (Probability: {probability:.3f})"
                '</div>',
                unsafe_allow_html=True
            )   

    if language=="French" :
        
        
        # ID du fichier sur Google Drive
        file_id = '1bS3R3rSyWLJswJI0M_qD1L1y-UGkNnjU'

        # URL de l'API Google Drive pour télécharger le fichier
        url = f'https://drive.google.com/uc?id={file_id}'

        # Utilisez la méthode `tf.keras.utils.get_file` pour charger le modèle directement depuis Google Drive
        model_path = tf.keras.utils.get_file('modele_french.h5', origin=url, extract=False, cache_subdir='/content/drive/MyDrive/model/')

        # Chargez le modèle
        model = tf.keras.models.load_model(model_path,compile=False)
        def my_reprocess(sentences, max_len=21):
                    X =  tokenizers.texts_to_sequences(sentences)

                        # Pad the sequences
                    X = pad_sequences(X, maxlen=max_len)

                    return X
        with open('./tokenizer_french.json', 'r', encoding='utf-8') as json_file:
                
                tokenizer_json = json.load(json_file)
                tokenizer = Tokenizer()
                tokenizers = tokenizer_from_json(tokenizer_json)
            
      
            
            
            #sentences = st.text_input("Enter text in French")
        processed_data = my_reprocess([sentences])

        #model = tf.keras.models.load_model('./model/modele_french.h5',compile=False)
        y_pred = model.predict(processed_data)

                # Interpret the prediction
        probability = y_pred[0][0]
        if probability >= 0.5 :
                    
            sentiment = "Positive sentiment"
        else:
            sentiment = "Negative"
       
        st.write(
                f'<div style="font-size: {font_size}; color: {text_color}; font-family: {font_family};">'
                f"Sentiment: {sentiment} (Probability: {probability:.3f})"
                '</div>',
                unsafe_allow_html=True
            )   
 
            