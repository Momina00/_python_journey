import streamlit as st
import pickle
import re
#LOAD SAVED MODEL &VECTORIZER
model = pickle.load(open("spam_model.pkl","rb"))
tfidf = pickle.load(open("vectorizer.pkl","rb"))

# cleaning function sam as training
def clean_text(text):
    text = text.lower()
    text = text.replace("subject","")
    text = re.sub(r"[^a-z0-9\s]","",text)
    return text
#PREDICTION FN SAME AS EARLIER
def predict_email(text):
     if len(text) < 6:
         return" Uncertain"
     text = text.lower()
     text = text.replace("subject","")
     text = re.sub(r"[^a-z0-9\s]","",text)
     vector = tfidf.transform([text])
     prediction = model.predict(vector)[0]

     if prediction == 1:
         return "Spam"
     else:
         return "Not-spam"

     #UI
st.title("___EMAIL SPAM DETECTOR___")
user_input = st.text_area("-------Enter your email here----------")
if st.button('Check'):
    if user_input.strip() != "":
        result = predict_email(user_input)
        st.success(result)
    else:
        st.warning("Please enter some text") 
