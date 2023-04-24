import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('Tweety.pkl', 'rb'))

st.title("Twitter Sentiment Analyzer")

input_tweet = st.text_area("Enter your tweet")

if st.button('Analyze'):

    # 1. preprocess
    transformed_tweet = transform_text(input_tweet)
    # 2. vectorize
    vector_input = vectorizer.transform([transformed_tweet])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Negative")
    else:
        st.header("Positive")
