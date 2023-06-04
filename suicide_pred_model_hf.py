# -*- coding: utf-8 -*-
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configurando a página
st.set_page_config(page_title="Predição de suícidio", page_icon=None, layout="centered", menu_items=None)
st.title('Essa pessoa precisa de ajuda?')
st.markdown('<div style="text-align: justify;">Com a tecnologia e redes sociais, informações pessoais são compartilhadas online. Modelo de AI pode analisar padrões linguísticos/comportamentais presentes nas postagens para prever risco de suicídio. A abordagem permite detecção precoce de sinais de alerta para intervenção rápida e prevenção da tragédia. Porém, não deve substituir avaliação clínica, mas ser uma ferramenta complementar. Aumenta eficácia dos esforços de prevenção do suicídio.</div>', unsafe_allow_html=True)
st.divider()
st.markdown('<div style="text-align: justify;">Para usar a ferramenta, escreva abaixo o texto para que o modelo possa retornar uma classifiação indicando se a pessoa precisa ou não de ajuda especializada:</div>', unsafe_allow_html=True)


#Inicializando o modelo pré-treainado
tokenizer = AutoTokenizer.from_pretrained('gooohjy/suicidal-bert')
model_hf = AutoModelForSequenceClassification.from_pretrained('gooohjy/suicidal-bert')

    
#Configurando a interação do usuário    
input_text = st.text_input('Insira o texto aqui', help='Insira um texto, ou trecho dele para avaliar se existe indícios suícida')


#Configurando a predição
tokens = tokenizer.encode(input_text, return_tensors='pt') 
result = model_hf(tokens)
result = int(torch.argmax(result.logits))
               
     
# "Rodando" a predição a partir do texto informado

if st.button('Obter resultado', help='Clique em "Obter resultado" para gerar uma avaliação do modelo de AI'):
   if (result == 1):
        st.warning('Essa pessoa potencialmente precisa de ajuda', icon="⚠️")
   else:
        st.success('Essa pessoa não apresenta indicios de risco', icon = "✅")