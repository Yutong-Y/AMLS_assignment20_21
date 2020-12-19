import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

import Tools as Tool
import A1 as A1
import A2 as A2
import B1 as B1
import B2 as B2

# ======================================================================================================================
# Data preprocessing
X_train_A1, X_val_A1, X_test_A1, y_train_A1, y_val_A1, y_test_A1 = Tool.data_preprocessingA(test_size=0.2, y_label='gender')
X_train_A2, X_val_A2, X_test_A2, y_train_A2, y_val_A2, y_test_A2 = Tool.data_preprocessingA(test_size=0.2, y_label='smiling')
X_train_B1, X_val_B1, X_test_B1, y_train_B1, y_val_B1, y_test_B1 = Tool.data_preprocessingB1(test_size=0.3)
X_train_B2, X_val_B2, X_test_B2, y_train_B2, y_val_B2, y_test_B2 = Tool.data_preprocessingB2(test_size=0.3)


# ======================================================================================================================
# Task A1
# Build and test LR model
y_pred_val_A1, y_pred_test_A1, A1_lr = A1.model_A1(X_train_A1, X_val_A1, X_test_A1, y_train_A1) 
acc_A1_train = accuracy_score(y_val_A1,y_pred_val_A1)
acc_A1_test = accuracy_score(y_test_A1,y_pred_test_A1)


# Detailed assessment:
#print('For validation set:')
#print(classification_report(y_val_A1,y_pred_val_A1))
#print('For test set:')
#print(classification_report(y_test_A1,y_pred_test_A1))
#plt = Tool.plot_learning_curve(A1_lr, X_train_A1, y_train_A1)       
#plt.show()
             
#del X_train_A1, X_test_A1, y_train_A1, y_test_A1, A1_lr   # Some code to free memory if necessary.

'''
----------The following code is used to train and tune the model---------
# For method1: Train the model with LogisticRegressionCV()
cv = 3     # same with 5
y_pred1, logreg = A1.logRegrCVPredict(X_train, y_train, X_val, cv)   #solver: liblinear
print('Accuracy on validation set: ' + str(accuracy_score(y_val,y_pred1)))
print(classification_report(y_val,y_pred1))

# For method2: Train the LR model with GridSearchCV()
y_pred2, logreg_cv = A1.logRegrPredictCV(x_train, y_train, x_val)
print('Accuracy on validation set: ' + str(accuracy_score(y_val,y_pred2)))
print(classification_report(y_val,y_pred2))

# Plot the learning curve
#plt = Tool.plot_learning_curve(logreg, X_train, y_train)      #For Method1
#plt = Tool.plot_learning_curve(logreg_cv, X_train, y_train)    #For Method2
'''

# ======================================================================================================================
# Task A2
y_pred_val_A2, y_pred_test_A2, A2_lr = A2.model_A2(X_train_A2, X_val_A2, X_test_A2, y_train_A2)
acc_A2_train = accuracy_score(y_val_A2,y_pred_val_A2)
acc_A2_test = accuracy_score(y_test_A2,y_pred_test_A2)

# Detailed assessment:
#print('For validation set:')
#print(classification_report(y_val_A2, y_pred_val_A2))
#print('For test set:')
#print(classification_report(y_test_A2,y_pred_test_A2))
#plt = Tool.plot_learning_curve(A2_lr, X_train_A2, y_train_A2)       
#plt.show()


#del X_train_A2, X_test_A2, y_train_A2, y_test_A2, A2_lr   # Some code to free memory if necessary.


'''
---------The following code is used to train and tune the models---------
# Train the Logistic Regression model with LogisticRegressionCV()
startTime = time.clock()
cv = 3
y_pred1, logreg = A2.logRegrPredict(X_train, y_train, X_val, cv)   #solver: liblinear
endTime = time.clock()
runTime = endTime - startTime
print('The running time is',runTime,'s')
print('Accuracy on validation set: ' + str(accuracy_score(y_val,y_pred1)))
print(classification_report(y_val,y_pred1))      
plt = Tool.plot_learning_curve(logreg, X_train, y_train)
plt.show()

# Train the SVM_CV model to find the optimized params
startTime = time.clock()
y_pred2, svmclf_cv = A2.svmPredictCV(X_train, y_train, X_val)
print(accuracy_score(y_val, y_pred2))
endTime = time.clock()
runTime = endTime - startTime
print('The running time is',runTime,'s')
print(classification_report(y_val,y_pred2)) 
plt = Tool.plot_learning_curve(svmclf_cv, X_train, y_train)
plt.show()

# Train the Random Forest model to find optimized params
startTime = time.clock()
y_pred3, rf_cv = A2.RanForCV(X_train, y_train, X_val)
endTime = time.clock()
runTime = endTime - startTime
print('The running time is',runTime,'s')
print(confusion_matrix(y_val, y_pred3))
print(classification_report(y_val,y_pred3)) 
plt = Tool.plot_learning_curve(rf_cv, X_train, y_train)
plt.show()
'''
# ======================================================================================================================
# Task B1
y_pred_val_B1, y_pred_test_B1, B1_svm = B1.model_B1(X_train_B1, X_val_B1, X_test_B1, y_train_B1)
acc_B1_train = accuracy_score(y_val_B1,y_pred_val_B1)
acc_B1_test = accuracy_score(y_test_B1,y_pred_test_B1)


# Detailed assessment:
#print('For validation set:')
#print(classification_report(y_val_B1, y_pred_val_B1))
#print('For test set:')
#print(classification_report(y_test_B1, y_pred_test_B1))
#plt = Tool.plot_learning_curve(B1_svm, X_train_B1, y_train_B1)       
#plt.show()

#del X_train_B1, X_test_B1, y_train_B1, y_test_B1, B1_svm   # Some code to free memory if necessary.


'''
---------The following code is used to train and tune the models---------
# Train the SVM model
startTime = time.clock()
y_pred, svmclf_cv = B1.svmPredictCV(X_train, y_train, X_val)
print(accuracy_score(y_val, y_pred1))

endTime = time.clock()
runTime = endTime - startTime
print('The running time is',runTime,'s')

# Build and test the RF model
startTime = time.clock()
y_pred_val2, y_pred_test2, B1_rf = B1.model_B1_rf(X_train, X_val, X_test, y_train) 
endTime = time.clock()
runTime = endTime - startTime
print('The running time is',runTime,'s')
print('Accuracy on validation set: ' + str(accuracy_score(y_val,y_pred_val2)))
print('Accuracy on test set: ' + str(accuracy_score(y_test,y_pred_test2)))
print('For validation set:')
print(classification_report(y_val,y_pred_val2))
print('For test set:')
print(classification_report(y_test,y_pred_test2)
'''
# ======================================================================================================================
# Task B2
y_pred_val_B2, y_pred_test_B2, B2_svm = B2.model_B2(X_train_B2, X_val_B2, X_test_B2, y_train_B2)
acc_B2_train = accuracy_score(y_val_B2,y_pred_val_B2)
acc_B2_test = accuracy_score(y_test_B2,y_pred_test_B2)


# Detailed assessment:
#print('For validation set:')
#print(classification_report(y_val_B2, y_pred_val_B2))
#print('For test set:')
#print(classification_report(y_test_B2, y_pred_test_B2))
#plt = Tool.plot_learning_curve(B2_svm, X_train_B1, y_train_B2)       
#plt.show()

#del X_train_B2, X_test_B2, y_train_B2, y_test_B2, B2_svm   # Some code to free memory if necessary.

'''
---------The following code is used to train and tune the models---------
# Train the SVM model to find optimal params
startTime = time.clock()
y_pred, svmclf_cv = svmPredictCV(X_train, y_train, X_val)
print(accuracy_score(y_val, y_pred))
endTime = time.clock()
runTime = endTime - startTime
print('The running time is',runTime,'s')   # >200s
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val,y_pred)) 
plt = Tool.plot_learning_curve(svmclf_cv, X_train, y_train)
plt.show()


# Train the RF model to find optimal params
startTime = time.clock()
y_pred, rf_cv = RanForCV(X_train, y_train, X_val)
endTime = time.clock()
runTime = endTime - startTime
print('The running time is',runTime,'s')
print(classification_report(y_val,y_pred)) 
plt = Tool.plot_learning_curve(rf_cv, X_train, y_train)
plt.show()
'''
# ======================================================================================================================
## Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))
