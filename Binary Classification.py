"""
-*- coding: utf-8 -*-

Titanic: Machine Learning from Disaster
Predicting survival on the Titanic 

For each PassengerId in the test set, this predict 0 or 1 value as a Survival prediction.

"""

import numpy as np
import pandas as pd

# Let's read file from local disk into dataframe titanic
titanic = pd.read_csv("MyCodes/titanic_survival_train.csv")

# information of titanic dataframe
print(titanic.info(),"\n")

# preprocessing data

# checking for null counts in every columns
isnull = titanic.isnull().sum()
print(isnull,"\n")

# Age, Cabin and Embarked features have got null counts
# Covering null values in Age column with it's mean value 
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].mean())

# Since, Cabin column has more than 75% of null values might not help in training a model 
# Let's drop this feature
titanic = titanic.drop('Cabin', axis = 1)

# Since, Embarked feature has 2 null counts. Let's drop 2 rows which help's removing these null rows
titanic = titanic.dropna()

# Moving further to convert categorical data to numerical
for each in ['Sex','Ticket','Embarked']:
    column = pd.Categorical.from_array(titanic[each])
    titanic[each] = column.codes
    
# features are up now to train the model 
features = ['Pclass','Sex','Age','SibSp','Ticket','Embarked','Fare']

# splitting the dataset into train/test in 75/25
margin = int(0.75 * titanic.shape[0])
train = titanic[:margin]
test = titanic[margin:]

# building logistic regression model
from sklearn.linear_model import LogisticRegression

# instantiating the model
lr = LogisticRegression()

# fitting the model
lr.fit(train[features], train['Survived'])

# predicting the result
logistic_prediction = lr.predict(test[features])

# checking the model accuracy
from sklearn.metrics import roc_auc_score

logistic_score = roc_auc_score(test['Survived'], logistic_prediction)
print("logistic regression :", logistic_score)
# achieving 0.7937 accuracy with logistic regression


# building decision tree classifier model and tuning them with hyper parameters
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(random_state=3, min_samples_split=2, min_samples_leaf=2)
dtc.fit(train[features], train['Survived'])
dtc_prediction = dtc.predict(test[features])
dtc_score = roc_auc_score(test['Survived'], dtc_prediction)
print("decision tree classifier :", dtc_score)
# achieving 0.8211 accuracy with decision tree classifier


# building random forest classifier model and tuning them with hyper parameters
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=250, random_state=3, min_samples_split=2, bootstrap = True, min_samples_leaf=3)
rfc.fit(train[features],train['Survived'])
rfc_prediction = rfc.predict(test[features])
rfc_score = roc_auc_score(test['Survived'], rfc_prediction)
print("random forest classifier :", rfc_score)
# achieving 0.8528 accuracy with random forest classifier

