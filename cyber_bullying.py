
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv('Reddit_Data.csv')

data.head()

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def preprocess(text):
  text=text.lower()
  tokens=word_tokenize(text)
  stopword=stopwords.words('english')
  filtered_tokens=[word for word in tokens if word not in stopword]
  limitizer=WordNetLemmatizer()
  lemmatized_tokens=[limitizer.lemmatize(word) for word in filtered_tokens]
  processed_text=' '.join(lemmatized_tokens)
  return processed_text

data['clean_comment']=data['clean_comment'].apply(str)

data['clean_comment']=data['clean_comment'].apply(preprocess)

X=data['clean_comment']
y=data['category']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.feature_extraction.text import TfidfVectorizer
tfv=TfidfVectorizer()
X_tfv=tfv.fit_transform(X_train)
X_test_tfv=tfv.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_tfv,y_train)

dt.predict(X_tfv)

from sklearn.metrics import accuracy_score
Y_pred=dt.predict(X_tfv)
accurecy_score=accuracy_score(y_train,Y_pred)
print(accurecy_score*100,"%")

inp=input("enter the tweet")
preprocessed_input=preprocess(inp)
vectorised_input=tfv.transform([preprocessed_input])
prediction=dt.predict(vectorised_input)
print(prediction)
if 0 in prediction:
  print("Neutral Tweet")
elif 1 in prediction:
  print("Positive Tweet")
else:
  print("Negative Tweet")

