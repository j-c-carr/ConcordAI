import os
import pandas as pd
import numpy as np
import sklearn as skl
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
from textblob import classifiers

import nltk

data = pd.read_csv('./data_with_bayes.csv')
target = []
for entry in data['pos_neg']:
	if entry == "pos":
		target.append(1)
	else:
		target.append(0)

bayes_predict=[]
for entry in data['bayes_prediction']:
	if entry == "pos":
		bayes_predict.append(1)
	else:
		bayes_predict.append(0)

data['target']=target
data['bayes_predict']=bayes_predict

X=data.iloc[:, [False, False, False, True, True, False, False, True]]
Y=data['target']

clf = RandomForestClassifier(n_estimators=100, max_depth = 4)
clf.fit(X, Y)

dump(clf, 'classifier.joblib')

