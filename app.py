import streamlit as st
# import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer=TfidfVectorizer()


loaded_model=pickle.load(open(r"model_1.csv","rb"))
vectorizer=pickle.load(open("vectorizer.csv","rb"))

def prediction(input):
    input=vectorizer.transform(input)
    input_data = input.toarray()
    # input_data = input_data.reshape(1, -1)
    prediction = loaded_model.predict(input_data)
    # print(prediction)
    if prediction==0:
        return "The person is in joy"
    elif prediction==1:
        return "The person is sad"

    elif prediction==2:
        return "The person is in anger"
    elif prediction==3:
        return "The person is in fear"
    elif prediction==5:
        return "The person is surprised"
    else:
        return "The person is in love"


#creatin main function
def main():

    st.title("Sentiment Analysis")
    text=st.text_input("Enter the text")


#initialising customer variable
    customer=""
#create a button
    if st.button("Predict"):
        customer=prediction([text])

    st.success(customer)

if __name__=="__main__":
    main()