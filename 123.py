# -*- coding: utf-8 -*-
"""
Created on Thu Nov 01 19:59:31 2018

@author: jacky
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 09:38:07 2018

@author: jacky
"""

#Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib






#Creating Dataset and including the first row by setting no header as input
dataset = pd.read_csv('C:\Users\jacky\.spyder\ictSamData.csv', header = None, encoding = "GB2312")
#Renaming the columns
dataset.columns = ['region','type','ProName','1','2','3','4','5','Result']


#Creating the dependent variable class
factor = pd.factorize(dataset['Result'])
dataset.Result = factor[0]
definitions = factor[1]


X = dataset.iloc[:,0:8].values
y = dataset.iloc[:,8].values


# Creating the Training and Test set from data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 21)


# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)


#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
#max_depth=None, max_features='auto', max_leaf_nodes=None,
#min_impurity_decrease=0.0, min_impurity_split=None,
#min_samples_leaf=1, min_samples_split=2,
#min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1,
#oob_score=False, random_state=42, verbose=0, warm_start=False)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Reverse factorize (converting y_pred from 0s and 1s to 无欠费 and 欠费
reversefactor = dict(zip(range(2),definitions))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)



X_pre = scaler.transform([[0,4,0,29600000,22414000,5963174,0,29600000]])
Y_pre = classifier.predict(X_pre)

if Y_pre[0] == 0:
    print ('该客户暂无欠费风险')
else:
    print ('该客户存在欠费风险')





