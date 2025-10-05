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

model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))


st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

# if st.button('Predict'):

#     # 1. preprocess
#     transformed_sms = transform_text(input_sms)
#     # 2. vectorize
#     vector_input = tfidf.transform([transformed_sms])
#     # 3. predict
#     result = model.predict(vector_input)[0]
#     # 4. Display
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    
    # 2. vectorize (convert to dense)
    vector_input = tfidf.transform([transformed_sms]).toarray()
    
    # 3. predict
    result = model.predict(vector_input)[0]
    
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")



# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer

# # ----------------------------
# # 🔹 Load the saved model & vectorizer
# # ----------------------------
# tfidf = pickle.load(open('vectorizer.pkl', 'rb'))   # Load trained TF-IDF vectorizer
# model = pickle.load(open('model.pkl', 'rb'))        # Load trained Naive Bayes model

# # ----------------------------
# # 🔹 Function to preprocess incoming SMS
# # ----------------------------
# import string
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer

# ps = PorterStemmer()

# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)
    
#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(ps.stem(i))

#     return " ".join(y)

# # ----------------------------
# # 🔹 Take user input and predict
# # ----------------------------
# input_sms = input("Enter the message: ")

# # Step 1: Transform text
# transformed_sms = transform_text(input_sms)

# # Step 2: Vectorize input using loaded TF-IDF
# vector_input = tfidf.transform([transformed_sms])

# # Step 3: Predict using loaded model
# result = model.predict(vector_input)[0]

# # ----------------------------
# # 🔹 Display the result
# # ----------------------------
# if result == 1:
#     print("\n🚨 Spam Message Detected!")
# else:
#     print("\n✅ Ham (Safe) Message Detected!")
