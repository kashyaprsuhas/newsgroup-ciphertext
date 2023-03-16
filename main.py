import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from pickle import load
import numpy as np
import pandas as pd

st.header("Newsgroup Ciphertext App")

data = pd.read_csv("https://raw.githubusercontent.com/kashyaprsuhas/newsgroup-ciphertext/main/dataset/test.csv")

# load model
model = Pipeline(memory=None, steps=[
        ('scaler', MaxAbsScaler(copy=False)),
        ('clf', LogisticRegression(multi_class='multinomial', verbose=2, n_jobs=-1))
    ])

# model.load_model("ciphertext_model.json")
model = load(open('ciphertext_model.pkl', 'rb'))

if st.checkbox('Show Test Dataset'):
    data
    
input_ciphertext = st.text_input("Enter Ciphertext: ", key="input_ciphertext")

input_ciphertext = [input_ciphertext]

if st.button('Make Prediction'):
    st.write(f"The given ciphertext belongs to the newsgroup: {input_ciphertext}")
    tfidf = load(open('tfidf.pkl', 'rb'))
    input_data_features = tfidf.transform(input_ciphertext)
    input_data_x = input_data_features.tocsr()
    del(tfidf)
    prediction = model.predict(input_data_x)
    print("final prediction", prediction)
    st.write(f"The given ciphertext belongs to the newsgroup: {prediction}")



