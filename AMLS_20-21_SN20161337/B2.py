# B2: Eye Color Classifier - SVM vs RF

# Import the required libraries

import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve

import time
import matplotlib.pyplot as plt
import pandas as pd

import Tools as Tool
import warnings
warnings.filterwarnings("ignore")


# =============Train the Support Vector Machine & Random Forest Model==================
# Build the SVM model with the optimized params found by GridSearchCV
def svmPredictCV(x_train, y_train, x_val):
    svmclf = svm.SVC()
    param_grid = [{'kernel': ['poly'], 'degree': [2,3], 'C': [1,10]}
                  ]
    grid_search = GridSearchCV(svmclf, param_grid, cv=3, scoring='accuracy')    
    grid_search.fit(x_train, y_train)
    
    # print dataframe  
    result = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]), pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["Accuracy"])], axis=1)
    print(result) 
    best_accuracy_score = grid_search.best_score_
    best_params = grid_search.best_params_
    print("Best: %f using %s" % (best_accuracy_score, best_params))
    
    # predict with the best classifier
    svmclf_cv = grid_search.best_estimator_
    y_pred = svmclf_cv.predict(x_val)
    
    return y_pred, svmclf_cv  


# Build Random Forest Model using Optimized n by GridSearchCV
def RanForCV(x_train, y_train, x_val):    
    rf = RandomForestClassifier(criterion='gini', max_features='log2', max_depth=10) #cv=3
    n_estimators = [10,30,50,100,200,300]
    param_grid = dict(n_estimators=n_estimators)
    grid_search = GridSearchCV(rf, param_grid)    
    grid_search.fit(x_train, y_train)    
    
    # print dataframe  
    result = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]), pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["Accuracy"])], axis=1)
    print(result) 
    best_accuracy_score = grid_search.best_score_
    best_params = grid_search.best_params_
    print("Best: %f using %s" % (best_accuracy_score, best_params))
    
    # predict with the best classifier
    rf_cv = grid_search.best_estimator_
    y_pred = rf_cv.predict(x_val)
    
    return y_pred, rf_cv  

# =============Model B1: Build SVM model with optimized params==================
# =====================Build RF model with optimized params====================
# Model_B2: Build the SVM model with the optimized params
def model_B2(x_train, x_val, x_test, y_train):
    B2_svm = svm.SVC(kernel='poly', C=1, degree=2)
    B2_svm.fit(x_train, y_train)
    y_pred_val = B2_svm.predict(x_val)
    y_pred_test = B2_svm.predict(x_test)
    
    return y_pred_val, y_pred_test, B2_svm

# Build Random Forest Model
def model_B2_rf(x_train, x_val, x_test, y_train):
    B2_rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='log2', max_depth=10) 
    B2_rf.fit(X_train, y_train)
    y_pred_val = B2_rf.predict(x_val)
    y_pred_test = B2_rf.predict(x_test)
    
    return y_pred_val, y_pred_test, B2_rf

