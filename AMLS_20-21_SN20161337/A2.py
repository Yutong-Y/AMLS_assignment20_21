# A2: Smiling Classifier - Random Forest vs Support Vector Machine vs Logistic Regression

# Import the required libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score

import time

import Tools as Tool
import warnings
warnings.filterwarnings("ignore")


# =============Train the Logistic Regression & Support Vector Machine & Random Forest Model==================
# Build Logistic Regression Model with Optimized C by Cross Validation 
def logRegrPredict(x_train, y_train, x_val, cv):    
    logreg = LogisticRegressionCV(solver='liblinear', cv=cv)   # default: L2 penalty
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_val)
    print('Optimized C is', logreg.C_)
    #print('accuracy score:',accuracy_score(y_val, y_pred))
    return y_pred, logreg 


# Build the SVM model with the optimized params found by GridSearchCV
def svmPredictCV(x_train, y_train, x_val):
    svmclf = svm.SVC()
    param_grid = [{'kernel': ['linear'], 'C': [1, 10, 100]},
                  {'kernel': ['rbf'], 'gamma': [1e-3, 1e-2], 'C': [1, 10, 100]},
                  {'kernel': ['poly'], 'degree': [2, 3], 'C': [1, 10, 100]}
                  ]
    grid_search = GridSearchCV(svmclf, param_grid, cv=3, scoring='accuracy',n_jobs=-1)  #, scoring='neg_log_loss'  
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
    n_estimators = [10,30,50,100,150,200]
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




# =============Model A2: Build LR model with optimized params==================
# ======================Build SVM model with optimized params==================
# =====================Build RF model with optimized params====================

# Model_A2: LR(solver='liblinear', C=0.006)
def model_A2(x_train, x_val, x_test, y_train):    
    A2_lr = LogisticRegression(solver='liblinear', C=0.006)   # default: L2 penalty
    A2_lr.fit(x_train, y_train)
    y_pred_val = A2_lr.predict(x_val)
    y_pred_test = A2_lr.predict(x_test) 

    return y_pred_val, y_pred_test, A2_lr 


# Model_A2_svm: Build the SVM model with the optimized params
def model_A2_svm(x_train, x_val, x_test, y_train):
    A2_svm = svm.SVC(kernel='linear', C=1)
    A2_svm.fit(x_train, y_train)
    y_pred_val = A2_svm.predict(x_val)
    y_pred_test = A2_svm.predict(x_test)
    
    return y_pred_val, y_pred_test, A2_svm 


# Model_A2_rf: Build the RF model with the optimized params
def model_A2_rf(x_train, x_val, x_test, y_train):
    A2_rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='log2', max_depth=10) 
    A2_rf.fit(X_train, y_train)
    y_pred_val = A2_rf.predict(x_val)
    y_pred_test = A2_rf.predict(x_test)
    
    return y_pred_val, y_pred_test, A2_rf

