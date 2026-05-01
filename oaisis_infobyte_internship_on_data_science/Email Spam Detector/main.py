import nltk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
# from nltk.corpus import stopwords 
# from ntlk.stem import PorterStemmer

df = pd.read_csv(r"C:\Users\hp\Desktop\_python_journey\oaisis_infobyte_internship_on_data_science\Email Spam Detector\spam_ham_dataset.csv")
df.drop('Unnamed: 0',axis=1,inplace=True)
df.columns = df.columns.str.strip().str.replace(" ","")
# DATA IS CLEANED ,SPACES REMOVED,AND NO COLUMNS CONTAINS NULL VALUES

#NOW COMES TEXT-PREPROCESSING(CORE STEP)
#THE DATASET USED IN THIS HAS PUNCTUATIONS,USELESS SPACES,SPECIAL CHARS.......
#SO WE NEED .. 1: TO LOWERCASE,,,2: REMOVE PUNCTUATION,3: REMOVE SPCL CHARS USING REGEX..
# FOR THIS DATASET, WE CAN SKIP STOPWORD REMOVAL AND STEMMING SINCE TEXTUAL DATA IS SMALL
def clean_text(text):
    text = text.lower()
    text = text.replace("subject","")
    text = re.sub(r"[^a-z0-9\s]","",text)
    # df.drop('text',axis=1,inplace=True)
    return text
df['clean_text'] = df['text'].apply(clean_text)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=3000 ,min_df = 2)

X = tfidf.fit_transform(df['clean_text'])
y = df['label_num']
# print(tfidf.get_feature_names_out()) --------> SHOWS ALL LEARNED FEATURES
# print(X.shape)------> (5496,5000)---> 5496 is 

#TRAIN-TEST-SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split( X, y, test_size =0.2, random_state = 42)


# MMODEL TRAINING
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train,y_train)

#prediction by model
y_pred = model.predict(X_test)

#MODEL EVALUATION
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# print("Accuracy: ",accuracy)
# print("Precision:  ",precision)
# print("recall: ", recall)
# print("Confusion_matrix:  \n",cm)
import pickle
pickle.dump(model , open("spam_model.pkl","wb"))
pickle.dump(tfidf, open("vectorizer.pkl","wb"))
    

#PREDICTION FUNCTION
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

# print(predict_email(" Kindly check the previous email Sarah."))


#OPTIONAL: PIPELINE
# from sklearn.pipeline import Pipeline
