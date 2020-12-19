# B1: Face Shape Classifier - SVM vs RF

# Import the required libraries

import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score

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
    param_grid = [{'kernel': ['linear'], 'C': [0.5,1,10,100]},
                  {'kernel': ['rbf'], 'gamma': [0.01], 'C': [1,3,5]},
                  {'kernel': ['poly'], 'degree': [2,3], 'C': [0.5,1,10,100]}
                  ]
    grid_search = GridSearchCV(svmclf, param_grid, cv=3, scoring='accuracy', n_jobs=-1) 
    grid_search.fit(x_train, y_train)
    
    # print result dataframe  
    result = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]), pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["Accuracy"])], axis=1)
    print(result) 
    best_accuracy_score = grid_search.best_score_
    best_params = grid_search.best_params_
    print("Best: %f using %s" % (best_accuracy_score, best_params))
    
    # predict with the best classifier
    svmclf_cv = grid_search.best_estimator_
    y_pred = svmclf_cv.predict(x_val)
    
    return y_pred, svmclf_cv  


# =============Model B1: Build SVM model with optimized params==================
# =====================Build RF model with optimized params====================
# Model_ B1: Build SVM model with optimized params
def model_B1(x_train, x_val, x_test, y_train):
    B1_svm = svm.SVC(kernel='rbf', C=1, gamma=0.01)
    B1_svm.fit(x_train, y_train)
    y_pred_val = B1_svm.predict(x_val)
    y_pred_test = B1_svm.predict(x_test)
    
    return y_pred_val, y_pred_test, B1_svm 

# Build Random Forest Model; tune the hyper-parameters manually
def model_B1_rf(x_train, x_val, x_test, y_train):
    B1_rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_features=0.3, max_depth=8)
    B1_rf.fit(X_train, y_train)
    y_pred_val = B1_rf.predict(x_val)
    y_pred_test = B1_rf.predict(x_test)
    
    return y_pred_val, y_pred_test, B1_rf 

