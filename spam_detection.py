

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv("spamorham.csv")

data.head()

data['Class'].value_counts()

data['Class']=data['Class'].replace(["ham","spam"],[1,0])

data.head()

import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
#nltk.download('punkt_tab')
#nltk.download('punkt')

def preprocess(text):
  text=text.lower()
  tokens=word_tokenize(text)
  stopword=stopwords.words('english')
  filtered_tokens=[word for word in tokens if word not in stopword]
  lemetizer=WordNetLemmatizer()
  lemitized_words=[lemetizer.lemmatize(word) for word in filtered_tokens]
  preprocessed_text=' '.join(lemitized_words)
  return preprocessed_text

data['sms']=data['sms'].apply(str)

data['sms'] = data['sms'].apply(preprocess)

X=data['sms']
y=data['Class']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X_train_cv=cv.fit_transform(X_train)
X_test_cv=cv.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train_cv,y_train)

rf.predict(X_test_cv)

# prompt: how to see accurecy

from sklearn.metrics import accuracy_score

y_pred = rf.predict(X_test_cv)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy*100)

# prompt: how to give user input to predect

user_input = input("Enter a message: ")
user_input_processed = preprocess(user_input)
user_input_cv = cv.transform([user_input_processed])
prediction = rf.predict(user_input_cv)[0]

if prediction == 1:
  print("Prediction: Ham")
else:
  print("Prediction: Spam")

'''
user_input=input("enter a message")
processed_inputs=preprocess(user_input)
inputss=cv.transform([processed_inputs])
predictions=rf.predict(inputss)[0]
if predictions==1:
  print("prediction:ham")
else:
  print("prediction:spam")
  '''

