import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

train = pd.read_csv("S_train.csv")
test = pd.read_csv("S_test.csv")

# Shrink file size

n = 10  # every n-th line
train = pd.read_csv("train.csv", header=0, skiprows=lambda i: i % n != 0)
train.to_csv('S_train.csv', index=False)
n=20
test = pd.read_csv("test.csv", header=0, skiprows=lambda i: i % n != 0)
test.to_csv('S_test.csv', index=False)


#convert category to numbers
event = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
experiment = {'CA':0,'DA':1,'SS':3,'LOFT':4}
test.experiment = [experiment[item] for item in test.experiment]
train.event = [event[item] for item in train.event]
train.experiment = [experiment[item] for item in train.experiment]




######### PCA ################
train_data = train.drop('event', axis = 1)
train_target = train['event']

print("train------")
print("shape: " ,train_data.shape)
print(train_data.head())
print("test-------")
print("shape: " ,train_target.shape)
print(train_target.head())
print(train_target.info)



scaler = StandardScaler()

#Fit on training set only.
scaler.fit(train)


#Apply transform to both the training set and the test set.
train_splt = scaler.transform(train_splt)
test_splt = scaler.transform(test_splt)

#save 95% of information
pca = PCA(.95)
pca.fit(train_splt)
print("num of component still exist: ", pca.n_components_)

#mapping of sets
train_splt = pca.transform(train_splt)
test_splt = pca.transform(test_splt)


#printing train and test head after converting category to int
x_train, x_test,y_train, y_test = train_test_split(train_data, train_target, test_size=0.2, random_state=0)
print("shape: " ,x_train.shape, y_train.shape)
print("shap test: " ,x_test.shape, y_test.shape)


print("shape before: " ,train.shape)
print(train.head())
print("-------------------------")
print("shape before: " ,test.shape)
print(test.head())
print("-------------------------")



############ CROSS VALIDATION ###################
train_data = train.drop('event', axis=1)
target = train['event']

print("-- CV --")
#CV- K = 10
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

#kNN
clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

# kNN Score
kn = round(np.mean(score)*100, 2)
print("-- KNN --", kn)
print("-----")

#Decision Tree
clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

# decision tree Score
dt = round(np.mean(score)*100, 2)
print("-- DT --", dt)
print("-----")

#Random Forest
clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

# Random Forest Score
rf = round(np.mean(score)*100, 2)
print("-- RF --", rf)
print("-----")

#Naive Bayes
clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

# Naive Bayes Score
nb = round(np.mean(score)*100, 2)
print("-- NB --", nb)
print("-----")

#SVM
clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

# SVM Score
sv = round(np.mean(score)*100,2)
print("-- SVC --", sv)
print("-----")

#Testing
max_m = max(kn,dt,rf,nb)
if max_m == kn:
    clf = KNeighborsClassifier(n_neighbors=13)
elif max_m == dt:
    clf = DecisionTreeClassifier()
elif max_m == rf:
    clf = RandomForestClassifier(n_estimators=13)
elif max_m == nb:
    clf = GaussianNB
#else:
#   clf = SVC()
print("-- max --", max_m)
print("-----")
print("-- predict --")

print("train_data head")
print(train_data.head())

print("target head")
print(target.head())

clf = KNeighborsClassifier(n_neighbors=13)
clf.fit(train_data, target)
prediction = clf.predict(test)

print("-- end of predict --")
print("-----")

submission = pd.DataFrame(np.concatenate((np.arange(len(test))[:, np.newaxis], prediction), axis=1), columns=['id', 'A', 'B', 'C', 'D'])
submission['id'] = submission['id'].astype(int)

print("sub head")
print(submission.head())

# select the columns that will be part of the submission
submission = test[['id', 'A', 'B', 'C', 'D']]
# save the submission dataframe as a csv file
submission.to_csv('submission.csv', index=False, columns=['id', 'A', 'B', 'C', 'D'])

print("-- End of sub --")
print("-----")

