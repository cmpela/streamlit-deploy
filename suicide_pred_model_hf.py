# -*- coding: utf-8 -*-
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configurando a página
st.set_page_config(page_title="Suicide Prediction", page_icon=None, layout="centered", menu_items=None)
st.title('Knows a person who need help?')
st.markdown('<div style="text-align: justify;">With technology and social media, personal information is shared online. AI model can analyze linguistic/behavioral patterns present in posts to predict suicide risk. The approach allows early detection of warning signs for rapid intervention and prevention of tragedy. However, it should not replace clinical assessment but be a complementary tool. It increases effectiveness of suicide prevention efforts.</div>', unsafe_allow_html=True)
st.divider()
st.markdown('<div style="text-align: justify;">To use the tool, type text below so that the model can return a rating indicating whether or not the person needs specialist help:</div>', unsafe_allow_html=True)


#Inicializando o modelo pré-treainado
tokenizer = AutoTokenizer.from_pretrained('gooohjy/suicidal-bert')
model_hf = AutoModelForSequenceClassification.from_pretrained('gooohjy/suicidal-bert')

    
#Configurando a interação do usuário    
input_text = st.text_input('Insert the text here', help='Enter a text, or an extract from it, to assess whether there is evidence of suicide.')


#Configurando a predição
tokens = tokenizer.encode(input_text, return_tensors='pt') 
result = model_hf(tokens)
result = int(torch.argmax(result.logits))
               
     
# "Rodando" a predição a partir do texto informado

if st.button('Get result', help='Click on "Get Result" to generate an evaluation of the IA model'):
   if (result == 1):
        st.warning('This person potentially needs help', icon="⚠️")
   else:
        st.success('This person presents no indications of risk', icon = "✅")