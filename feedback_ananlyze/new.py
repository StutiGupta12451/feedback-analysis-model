import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
model=load_model('feedback_ananlyze/feedback_analysis.h5', compile=False)
vectorizer=joblib.load("feedback_ananlyze/vectorizer.pkl")
def func(arr):
    ans=model.predict(arr)
    pred=(ans>0.5).astype("int32")
    if pred==0:
        return "negative"
    elif pred==1:
        return "positive"
    else:
        return "error"
st.title("Feedback Sentiment Analysis")
feed = st.text_input("Enter the feedback")
st.success("86% percent accuracy of model")
if st.button('process'):
    if not feed.strip():
        st.warning("Please Enter some text!!!!")
    else:
        feed=feed.lower()
        arr=vectorizer.transform([feed]).toarray()
        ans=func(arr)
        if ans=="negative":
            st.warning(ans)
        elif ans=="positive":
            st.success(ans)
        else:

            st.warning(ans)


