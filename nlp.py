# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 18:01:24 2018

@author: Somesh
"""
#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv("Restaurant_Reviews.tsv",delimiter="\t",quoting=3)

#Cleaning the texts
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
    review=re.sub("[^a-zA-Z]"," ",dataset["Review"][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review= [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
    review=" ".join(review)
    corpus.append(review)
    
#Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values

#using naive bayes classification model(73%)
"""#splitting the dataset
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

#fitting the naive bayes into the dataset
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)

#predicting the new results
pred=classifier.predict(X_test)

#computing the accuracy using confusion matrix function
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,pred)"""


"""#using decision tree classification model(71%)
#splitting the dataset
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

#fitting the decision tree classification into the dataset
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="entropy",random_state=0)
classifier.fit(X_train,y_train)

#predicting the new results
pred=classifier.predict(X_test)

#computing the accuracy using the confusion matrix function
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,pred)"""

#using random forest classification model(76%)
#splitting the dataset into training and testing units
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

#fitting the decision tree classification into the dataset
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=0)
classifier.fit(X_train,y_train)

#predicting the new results
pred=classifier.predict(X_test)

#computing the accuracy using the confusion matrix function
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,pred)

