

import pandas as pd
import numpy as np
import seaborn as se
import matplotlib.pyplot as plt
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data=pd.read_csv('movies.csv')

data.head()

data.shape

#selecting the relavent features for recomendation
selected_features=['genres','keywords','tagline','cast','director',]
print(selected_features)

#replaceing the null values with null string
for feature in selected_features:
  data[feature]=data[feature].fillna('')

#combining all the selected features
combined_features=data['genres']+' '+data['keywords']+' '+data['tagline']+' '+data['cast']+' '+data['director']

print(combined_features)

# converting the text data into feature vectors
vectorizer=TfidfVectorizer()

feature_vector=vectorizer.fit_transform(combined_features)

print(feature_vector)

#getting the similiraty score using cosine similiraty
similarity=cosine_similarity(feature_vector)

print(similarity) # 1 exact matches 0.72 indicates 72% match

similarity.shape

#getting a movie name from the user
movie_name=input('enter your favourate movie name : ')

# creating a list with all the movie names given in the data fram
titles_list=data['title'].tolist()
print(titles_list)

find_close_match=difflib.get_close_matches(movie_name,titles_list)
print(find_close_match)

close_match=find_close_match[0]
print(close_match)

# finding index of the movie with title
index_of_the_movie=data[data.title==close_match]['index'].values[0]
print(index_of_the_movie)
'''index_of_the_movie = None
for i in data.index:
  if data.title[i] == close_match:
    index_of_the_movie = i
    break

print(index_of_the_movie) # you can even use this'''

# getting a list of similar movies
similarity_score=list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)

# sorting the movies based on similarity score
sorted_list=sorted(similarity_score,key=lambda x:x[1],reverse=True) # (0, 0.033570748780675445)  x[0]= 0, x[1] = 0.033570748780675445
print(sorted_list)

# printing the names of similar movies based on index
print("The movies suggested for you : \n")
i=1
for movie in sorted_list:
  index=movie[0]  # movie[0] suggests 68 from this (68, 1.0000000000000002)
  Title=data[data.index==index]['title'].values[0]  # if data.index is equal to index in dataset then return title
  if(i<30):
    print(i,'',Title)
    i+=1

movie_name=input('enter your favourate movie name : ')
titles_list=data['title'].tolist()
find_close_match=difflib.get_close_matches(movie_name,titles_list)
close_match=find_close_match[0]
index_of_the_movie=data[data.title==close_match]['index'].values[0]
similarity_score=list(enumerate(similarity[index_of_the_movie]))
sorted_list=sorted(similarity_score,key=lambda x:x[1],reverse=True) # (0, 0.033570748780675445)  x[0]= 0, x[1] = 0.033570748780675445
print("The movies suggested for you : \n")
i=1
for movie in sorted_list:
  index=movie[0]  # movie[0] suggests 68 from this (68, 1.0000000000000002)
  Title=data[data.index==index]['title'].values[0]
  if(i<30):
    print(i,'',Title)
    i+=1

