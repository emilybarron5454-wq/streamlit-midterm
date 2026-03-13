import streamlit as st
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

st.title("Text Classification App")

st.write("This app demonstrates multiple classifiers for text classification.")

classifier = st.selectbox(
    "Choose classifier",
    ["Naive Bayes", "SVM", "Random Forest", "Decision Tree"]
)

train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

if classifier == "Naive Bayes":
    model = MultinomialNB()

elif classifier == "SVM":
    model = SGDClassifier()

elif classifier == "Random Forest":
    model = RandomForestClassifier()

elif classifier == "Decision Tree":
    model = DecisionTreeClassifier()

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', model),
])

pipeline.fit(train.data, train.target)

pred = pipeline.predict(test.data)

accuracy = np.mean(pred == test.target)

st.write("Accuracy:", accuracy)
