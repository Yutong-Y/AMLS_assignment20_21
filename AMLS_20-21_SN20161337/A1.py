# A1: Gender Classifier - Logistic Regression

# Import the required libraries
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

# =============Train the Logistic Regression Model to find the optimized params==================
# Method1: Build Logistic Regression Model with Optimized C by Cross Validation 
def logRegrCVPredict(x_train, y_train, x_val, cv):    
    logreg = LogisticRegressionCV(solver='liblinear', cv=cv)   # default: L2 penalty
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_val)
    print('Optimized C is', logreg.C_)

    return y_pred, logreg

# Method2: Find Optimized C for Logistic Regression Model by GridSearchCV
def logRegrPredictCV(x_train, y_train, x_val):    
    logreg = LogisticRegression(solver='liblinear')   # default: L2 penalty
    C = [0.001, 0.004, 0.04, 0.04, 0.1, 0.4, 1, 4, 10]
    param_grid = dict(C=C)
    grid_search = GridSearchCV(logreg, param_grid, cv=3, scoring='accuracy')    
    grid_search.fit(x_train, y_train)
    
    # print dataframe  
    result = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]), pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["Accuracy"])], axis=1)
    print(result)
    best_accuracy_score = grid_search.best_score_
    best_params = grid_search.best_params_
    print("Best: %f using %s" % (best_accuracy_score, best_params))
    
    # predict with the best classifier
    logreg_cv = grid_search.best_estimator_
    y_pred = logreg_cv.predict(x_val)
    
    return y_pred, logreg_cv 


# =============Model A1: Build the Logistic Regression Model with optimized params==================
# Model_A1: LR(solver='liblinear', C=0.04)
def model_A1(x_train, x_val, x_test, y_train):
    A1_lr = LogisticRegression(solver='liblinear', C=0.04)
    A1_lr.fit(x_train, y_train)
    y_pred_val = A1_lr.predict(x_val)
    y_pred_test = A1_lr.predict(x_test)
    
    return y_pred_val, y_pred_test, A1_lr
