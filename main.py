
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
import re
import streamlit as st

tf_idf = pickle.load(open('tf_idf.pkl', 'rb'))
model = pickle.load(open('Email_spam_classifier_model.pkl', 'rb'))

def text_cleaning(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if word not in set(stopwords.words('english'))]
    return ' '.join(text)

st.title('Email Spam Classifier')
input_area = st.text_area('enter the mail here to check for the spam')

if st.button('Predict'):
    transform = text_cleaning(input_area)
    vectors = tf_idf.transform([transform]).toarray()
    prediction = model.predict(vectors)[0]

    if prediction == 1:
        st.header('Spam')
    else:
        st.header('Ham')

value = 'default'
if st.button('Refresh'):
    value = ' '