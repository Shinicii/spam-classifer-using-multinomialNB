import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

tfidf = pickle.load(open("vectorizer.pkl", "rb"), )
model = pickle.load(open("mnbmodel.pkl", "rb"))

###fontend
st.title('Spam Detector')
input_txt = st.text_area("Enter text to classify")

####backend
ps = PorterStemmer()

def clean_text(message):

    message = message.lower()
    message = nltk.word_tokenize(message)

    cln = []
    for i in message:
        if i.isalnum():
            cln.append(i)

    message = cln[:]
    cln.clear()

    for i in message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            cln.append(i)
    message = cln[:]
    cln.clear()
    for i in message:
        cln.append(ps.stem(i))

    return " ".join(cln)


if st.button('Check'):
    text = clean_text(input_txt)
    vect_inp = tfidf.transform([text])

    result = model.predict(vect_inp)[0]

    if result == 1:
        st.subheader("SPAM Alert!")
    else:
        st.subheader("Not SPAM")
