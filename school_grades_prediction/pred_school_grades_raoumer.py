# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13, 2016

@author: RaoUmer
@E-mail: engr.raoumer943@gmail.com
"""
# importing the required modules

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.svm	import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler 
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score

import matplotlib.pyplot as plt
import seaborn as sns

#################################################################################################################
# Loading train and test dataset

# reading the dataset using pandas
data_train = pd.read_csv("traindata.csv")
data_test = pd.read_csv("testfeatures.csv")


print "Train data:", data_train.shape
print "Test data:", data_test.shape

#print data_train.describe()
#print data_train.head()

#################################################################################################################
# Pre-processing the Dataset

# Pre-processing the Training Dataset
# handling Categorical variables with two categories
le = LabelEncoder()
data_train['school'] = le.fit_transform(data_train['school'].values)
data_train['sex'] = le.fit_transform(data_train['sex'].values)
data_train['address'] = le.fit_transform(data_train['address'].values)
data_train['famsize'] = le.fit_transform(data_train['famsize'].values)
data_train['Pstatus'] = le.fit_transform(data_train['Pstatus'].values)
data_train['schoolsup'] = le.fit_transform(data_train['schoolsup'].values)
data_train['famsup'] = le.fit_transform(data_train['famsup'].values)
data_train['paid'] = le.fit_transform(data_train['paid'].values)
data_train['activities'] = le.fit_transform(data_train['activities'].values)
data_train['nursery'] = le.fit_transform(data_train['nursery'].values)
data_train['higher'] = le.fit_transform(data_train['higher'].values)
data_train['internet'] = le.fit_transform(data_train['internet'].values)
data_train['romantic'] = le.fit_transform(data_train['romantic'].values)

# handling Categorical variables with more than two categories
dummies_train = pd.get_dummies(data_train[['Mjob', 'Fjob', 'reason', 'guardian']])
data_train = pd.concat([data_train, dummies_train], axis=1)

# Pre-processing the Testing Dataset
# handling Categorical variables with two categories
data_test['school'] = le.fit_transform(data_test['school'].values)
data_test['sex'] = le.fit_transform(data_test['sex'].values)
data_test['address'] = le.fit_transform(data_test['address'].values)
data_test['famsize'] = le.fit_transform(data_test['famsize'].values)
data_test['Pstatus'] = le.fit_transform(data_test['Pstatus'].values)
data_test['schoolsup'] = le.fit_transform(data_test['schoolsup'].values)
data_test['famsup'] = le.fit_transform(data_test['famsup'].values)
data_test['paid'] = le.fit_transform(data_test['paid'].values)
data_test['activities'] = le.fit_transform(data_test['activities'].values)
data_test['nursery'] = le.fit_transform(data_test['nursery'].values)
data_test['higher'] = le.fit_transform(data_test['higher'].values)
data_test['internet'] = le.fit_transform(data_test['internet'].values)
data_test['romantic'] = le.fit_transform(data_test['romantic'].values)

# handling Categorical variables with more than two categories
dummies_test = pd.get_dummies(data_test[['Mjob', 'Fjob', 'reason', 'guardian']])
data_test = pd.concat([data_test, dummies_test], axis=1)

# After per-processing the training data by converting all categorical variables into numerical form
feature_cols_train = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
                'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup','paid', 'activities', 
                'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime',
                'goout', 'Dalc', 'Walc', 'health', 'absences',
                'Mjob_at_home', 'Mjob_other', 'Mjob_teacher', 'Mjob_health', 'Mjob_services',
                'Fjob_at_home', 'Fjob_other', 'Fjob_teacher', 'Fjob_health', 'Fjob_services',
                'reason_course', 'reason_home', 'reason_reputation', 'reason_other',           
                'guardian_father', 'guardian_mother', 'guardian_other'                
                ]
                
# After per-processing the testing data by converting all categorical variables into numerical form
feature_cols_test = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
                'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup','paid', 'activities', 
                'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime',
                'goout', 'Dalc', 'Walc', 'health', 'absences',
                'Mjob_at_home', 'Mjob_other', 'Mjob_teacher', 'Mjob_health', 'Mjob_services',
                'Fjob_at_home', 'Fjob_other', 'Fjob_teacher', 'Fjob_health', 'Fjob_services',
                'reason_course', 'reason_home', 'reason_reputation', 'reason_other',           
                'guardian_father', 'guardian_mother', 'guardian_other'                
                ]

# important features

feature_cols_train1 = ['school', 'sex', 'age', 'address', 'famsize', 'Medu', 'Fedu',
                'traveltime', 'studytime', 'failures', 'schoolsup', 'activities', 
                'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime',
                'Dalc', 'Walc', 'health', 'absences',
                'Mjob_at_home', 'Mjob_other', 
                'Fjob_at_home', 'Fjob_other',
                'reason_course',  'reason_reputation', 'reason_other',           
                'guardian_father'                
                ]

feature_cols_test1 = ['school', 'sex', 'age', 'address', 'famsize', 'Medu', 'Fedu',
                'traveltime', 'studytime', 'failures', 'schoolsup', 'activities', 
                'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime',
                 'Dalc', 'Walc', 'health', 'absences',
                'Mjob_at_home', 'Mjob_other', 
                'Fjob_at_home', 'Fjob_other', 
                'reason_course', 'reason_reputation', 'reason_other',           
                'guardian_father'              
                ]

#################################################################################################################
## EDA (Exploratory Data Analysis)
#
## visualizing the training dataset for Exploratory Data Analysis
#plt.figure(121)
#sns.set(style='whitegrid', context='notebook')
#sns.pairplot(data_train[feature_cols_train], size=2.5)
#plt.tight_layout()
#plt.savefig('data_train_eda.png', dpi=300)
#plt.show()
#
## plotting correlation matrix for training dataset
cm_train = np.corrcoef(data_train[feature_cols_train].values.T)
print "Train Corrcoef:", cm_train

#plt.figure(122)
#sns.set(font_scale=1.5)
#hm = sns.heatmap(cm_train, 
#            cbar=True,
#            annot=True, 
#            square=True,
#            fmt='.2f',
#            annot_kws={'size': 15},
#            yticklabels=feature_cols_train,
#            xticklabels=feature_cols_train)
#            
#plt.tight_layout()
#plt.savefig('corr_mat_train.png', dpi=300)
#plt.show()
#
#
## visualizing the testing dataset for Exploratory Data Analysis
#plt.figure(123)
#sns.set(style='whitegrid', context='notebook')
#sns.pairplot(data_test[feature_cols_test], size=2.5)
#plt.tight_layout()
#plt.savefig('data_test_eda.png', dpi=300)
#plt.show()
#
## plotting correlation matrix for testing dataset
cm_test = np.corrcoef(data_test[feature_cols_test].values.T)
print "Test Corrcoef:", cm_test
#plt.figure(124)
#sns.set(font_scale=1.5)
#hm = sns.heatmap(cm, 
#            cbar=True,
#            annot=True, 
#            square=True,
#            fmt='.2f',
#            annot_kws={'size': 15},
#            yticklabels=feature_cols_test,
#            xticklabels=feature_cols_test)
#            
#plt.tight_layout()
#plt.savefig('corr_mat_test.png', dpi=300)
#plt.show()

#################################################################################################################
# Training Phase

# create training data (Xtr) and its labels(ytr)
Xtr = data_train[feature_cols_train]
ytr = data_train.GRADE

# Applying Cross validation
X_train, X_test, y_train, y_test =	 train_test_split(Xtr, ytr, test_size=0.30, random_state=101)

# standardize the data
X_train_std = StandardScaler().fit_transform(X_train)
X_test_std = StandardScaler().fit_transform(X_test)
y_train_std = StandardScaler().fit_transform(y_train)
y_test_std = StandardScaler().fit_transform(y_test)

np.mean(X_train_std)
np.std(X_train_std)
# Applying SVR(linear kernel) without standardization
regr_svr_lin = SVR(kernel='linear', C=0.1, epsilon=1)
regr_svr_lin.fit(X_train, y_train)
y_train_pred_svr_lin = regr_svr_lin.predict(X_train)
y_test_pred_svr_lin = regr_svr_lin.predict(X_test)
print "SVR coef:", regr_svr_lin.coef_

# Applying SVR(linear kernel) with standardization
clf_svr_std_lin = SVR(kernel='linear', C=0.1, epsilon=1)
clf_svr_std_lin.fit(X_train_std, y_train_std)
y_train_pred_svr_std_lin = clf_svr_std_lin.predict(X_train_std)
y_test_pred_svr_std_lin = clf_svr_std_lin.predict(X_test_std)
#print('Slope: %.3f' % clf_svr.coef_)
#print('Intercept: %.3f' % clf_svr_std.intercept_)

# Applying SVR(Polynomial kernel) without standardization
clf_svr_poly = SVR(kernel='poly', C=1, degree=1, epsilon=1)
clf_svr_poly.fit(X_train, y_train)
y_train_pred_svr_poly = clf_svr_poly.predict(X_train)
y_test_pred_svr_poly = clf_svr_poly.predict(X_test)

# Applying SVR(Polynomial kernel) with standardization
clf_svr_std_poly = SVR(kernel='poly', C=1, degree=1, epsilon=1)
clf_svr_std_poly.fit(X_train_std, y_train_std)
y_train_pred_svr_std_poly = clf_svr_std_poly.predict(X_train_std)
y_test_pred_svr_std_poly = clf_svr_std_poly.predict(X_test_std)

# Applying SVR(rbf kernel) without standardization
clf_svr_rbf = SVR(kernel='rbf', C=100, gamma=0.0001, epsilon=1)
clf_svr_rbf.fit(X_train, y_train)
y_train_pred_svr_rbf = clf_svr_rbf.predict(X_train)
y_test_pred_svr_rbf = clf_svr_rbf.predict(X_test)

# Applying SVR(rbf kernel) with standardization
clf_svr_std_rbf = SVR(kernel='rbf', C=100, gamma=0.0001, epsilon=1)
clf_svr_std_rbf.fit(X_train_std, y_train_std)
y_train_pred_svr_std_rbf = clf_svr_std_rbf.predict(X_train_std)
y_test_pred_svr_std_rbf = clf_svr_std_rbf.predict(X_test_std)

# Computing MAE and MSE for Support Vector Regression without standardization
print "Without Standardization"
print('SVR(lin.): MAE train: %.3f, test: %.3f' % (mean_absolute_error(y_train, y_train_pred_svr_lin), mean_absolute_error(y_test, y_test_pred_svr_lin)))
print('SVR(lin.): MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred_svr_lin), mean_squared_error(y_test, y_test_pred_svr_lin)))

print('SVR(poly): MAE train: %.3f, test: %.3f' % (mean_absolute_error(y_train, y_train_pred_svr_poly), mean_absolute_error(y_test, y_test_pred_svr_poly)))
print('SVR(poly): MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred_svr_poly), mean_squared_error(y_test, y_test_pred_svr_poly)))

print('SVR(rbf): MAE train: %.3f, test: %.3f' % (mean_absolute_error(y_train, y_train_pred_svr_rbf), mean_absolute_error(y_test, y_test_pred_svr_rbf)))
print('SVR(rbf): MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred_svr_rbf), mean_squared_error(y_test, y_test_pred_svr_rbf)))

print "With Standardization"
# Computing MAE and MSE for Support Vector Regression with standardization
print('SVR(lin.): MAE train: %.3f, test: %.3f' % (mean_absolute_error(y_train_std, y_train_pred_svr_std_lin), mean_absolute_error(y_test_std, y_test_pred_svr_std_lin)))
print('SVR(lin.): MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train_std, y_train_pred_svr_std_lin), mean_squared_error(y_test_std, y_test_pred_svr_std_lin)))

print('SVR(poly): MAE train: %.3f, test: %.3f' % (mean_absolute_error(y_train_std, y_train_pred_svr_std_poly), mean_absolute_error(y_test_std, y_test_pred_svr_std_poly)))
print('SVR(poly): MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train_std, y_train_pred_svr_std_poly), mean_squared_error(y_test_std, y_test_pred_svr_std_poly)))

print('SVR(rbf): MAE train: %.3f, test: %.3f' % (mean_absolute_error(y_train_std, y_train_pred_svr_std_rbf), mean_absolute_error(y_test_std, y_test_pred_svr_std_rbf)))
print('SVR(rbf): MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train_std, y_train_pred_svr_std_rbf), mean_squared_error(y_test_std, y_test_pred_svr_std_rbf)))

# computing r^2 score
print "R^2 without standardization"
print('SVR(lin.): R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred_svr_lin), r2_score(y_test, y_test_pred_svr_lin)))
print('SVR(poly): R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred_svr_poly), r2_score(y_test, y_test_pred_svr_poly)))
print('SVR(rbf): R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred_svr_rbf), r2_score(y_test, y_test_pred_svr_rbf)))

print "R^2 with standardization"
print('SVR(lin.): R^2 train: %.3f, test: %.3f' % (r2_score(y_train_std, y_train_pred_svr_std_lin), r2_score(y_test_std, y_test_pred_svr_std_lin)))
print('SVR(poly): R^2 train: %.3f, test: %.3f' % (r2_score(y_train_std, y_train_pred_svr_std_poly), r2_score(y_test_std, y_test_pred_svr_std_poly)))
print('SVR(rbf): R^2 train: %.3f, test: %.3f' % (r2_score(y_train_std, y_train_pred_svr_std_rbf), r2_score(y_test_std, y_test_pred_svr_std_rbf)))

#################################################################################################################
#  Optimal Parameter selection

#cross validation on lin. SVR
scores = cross_val_score(estimator=regr_svr_lin, X=Xtr, y=ytr, cv=10).mean()
print "SVR CV scores:", scores

# parameter selection for SVR lin.
pipe_svr = Pipeline([('scl', StandardScaler()), ('clf', SVR())])

param_range_c = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
param_range_e = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

param_grid = {'clf__C': param_range_c, 'clf__epsilon': param_range_e, 'clf__kernel': ['linear']}

gs_lin = GridSearchCV(estimator=pipe_svr, param_grid=param_grid, cv=10)
gs_lin = gs_lin.fit(X_train, y_train)
print gs_lin.best_score_
print gs_lin.best_params_

# parameter selection for SVR poly
pipe_poly = Pipeline([('scl', StandardScaler()), ('clf', SVR())])

param_range_c = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
param_range_e = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
param_range_d = [1.0, 2.0, 3.0, 4.0, 5.0]

param_grid = {'clf__C': param_range_c, 'clf__epsilon': param_range_e, 'clf__degree': param_range_d, 'clf__kernel': ['poly']}

gs_poly = GridSearchCV(estimator=pipe_poly, param_grid=param_grid, cv=10)
gs_poly = gs_poly.fit(X_train, y_train)
print gs_poly.best_score_
print gs_poly.best_params_

# parameter selection for SVR rbf
pipe_rbf = Pipeline([('scl', StandardScaler()), ('clf', SVR())])

param_range_c = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
param_range_e = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

param_grid = {'clf__C': param_range_c, 'clf__epsilon': param_range_e, 'clf__gamma': param_range_c, 'clf__kernel': ['rbf']}

gs_rbf = GridSearchCV(estimator=pipe_rbf, param_grid=param_grid, cv=10)
gs_rbf = gs_rbf.fit(X_train, y_train)
print gs_rbf.best_score_
print gs_rbf.best_params_

# cross validation on lasso
regr_lassocv = linear_model.LassoCV(cv=10)
regr_lassocv.fit(Xtr, ytr)

print "lasso alpha:", regr_lassocv.alpha_

# cross validation on elasticnet
regr_elasticcv = linear_model.ElasticNetCV(cv=10)
regr_elasticcv.fit(Xtr, ytr)

print "ElasticNet alpha:", regr_elasticcv.alpha_
print "ElasticNet l1 ratio:", regr_elasticcv.l1_ratio_

#################################################################################################################
## Feature Selection
# Applying Lasso without standardization
regr_lasso = linear_model.Lasso(alpha=.03)
regr_lasso.fit(X_train, y_train)
y_train_pred_svr_lasso = regr_lasso.predict(X_train)
y_test_pred_svr_lasso = regr_lasso.predict(X_test)
print "Lasso coef:", regr_lasso.coef_
print "Feature coef:",zip(feature_cols_train, regr_lasso.coef_)

# Applying ElasticNet without standardization
regr_elasticnet = linear_model.ElasticNet(alpha=.03, l1_ratio=.5)
regr_elasticnet.fit(X_train, y_train)
y_train_pred_svr_elasticnet = regr_elasticnet.predict(X_train)
y_test_pred_svr_elasticnet = regr_elasticnet.predict(X_test)
print "ElasticNet coef:", regr_elasticnet.coef_
print "Feature coef:",zip(feature_cols_train, regr_elasticnet.coef_)

print('Lasso: MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred_svr_lasso), mean_squared_error(y_test, y_test_pred_svr_lasso)))
print('ElasticNet: MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred_svr_elasticnet), mean_squared_error(y_test, y_test_pred_svr_elasticnet)))

print('Lasso: R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred_svr_lasso), r2_score(y_test, y_test_pred_svr_lasso)))
print('ElasticNet: R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred_svr_elasticnet), r2_score(y_test, y_test_pred_svr_elasticnet)))

##########################################################################################################################################################################
#Testing Phase

# create testing data (Xtt)
Xtt = data_test[feature_cols_test]

# Applying SVR(linear kernel) 
regr_svr_lin = SVR(kernel='linear', C=0.1, epsilon=1)
regr_svr_lin.fit(Xtr, ytr)
ytt_test_pred_svr_lin = regr_svr_lin.predict(Xtt)

output_svr = pd.DataFrame(ytt_test_pred_svr_lin)
output_svr.to_csv('output_svrlin.csv')

# Applying Lasso
regr_lasso = linear_model.Lasso(alpha=.03)
regr_lasso.fit(Xtr, ytr)
ytt_test_pred_lasso = regr_lasso.predict(Xtt)

output_lasso = pd.DataFrame(ytt_test_pred_lasso)
output_lasso.to_csv('output_lasso.csv')

# Applying ElasticNet 
regr_elasticnet = linear_model.ElasticNet(alpha=.03, l1_ratio=.5)
regr_elasticnet.fit(Xtr, ytr)
ytt_test_pred_elasticnet = regr_elasticnet.predict(Xtt)

output_elasticnet = pd.DataFrame(ytt_test_pred_lasso)
output_elasticnet.to_csv('output_elasticnet.csv')
