# README
## Brief description of the organization of the project:
A1: Gender classification  
Dataset: celeba, celeba_test  
Test algorithms: Logistic Regression  
Final model: Logistic Regression

A2: Emotion classification  
Dataset: celeba, celeba_test  
Test algorithms: Logistic Regression, Support Vector Machine, Random Forest  
Final model: Logistic Regression

B1: Face shape classification  
Dataset: cartoon_set, cartoon_set_test  
Test algorithms: Support Vector Machine, Random Forest  
Final model: Support Vector Machine 

B2: Eye color classification  
Dataset: cartoon_set, cartoon_set_test  
Test algorithms: Support Vector Machine, Random Forest  
Final model: Support Vector Machine  
    
## The role of each file:
All the project files are in 'AMLS_20-21_SN20161337'.

main.py:  
The main funtion file, used for four fine-tuned models' training and testing. The basic purpose is to print accuracy rate for the trained models, but you can uncomment some lines to see detailed assessment or training codes.

A1.py; A2.py; B1.py; B2.py:  
The funtions to build all mentioned models for each task.

Tools.py:  
The functions to perform data preprocessing for each task, and the function to plot the learning curve graph.

features_extraction_A.py; features_extraction_B.py; shape_predictor_68_face_landmarks.dat:  
The functions and materials to extract image features for taskA1/A2 and taskB1/B2.  
NOTE: Since github limits the size of file to be uploaded, I haven't solve this problem of adding 'shape_predictor_68_face_landmarks.dat' file into the repertory. Please add it when you run the codes! Thank you!

---------------------------------------------

A1.ipynb; A2.ipynb; B1.ipynb; B2.ipynb:  
The well-organized jupyter notebook files that can be runned seperately to see the whole process for building, training, tuning, and testing models for each task.

A1_LR.ipynb; A2_RF&LR&SVM.ipynb; B1_SVM&RF.ipynb; B2_RF&SVM.ipynb:  
The original version of testing and debugging models. Most intermediate experiment results such as hyper-parameters testing and tuning could be found in those files.

## The packages required to run the code:
numpy; matplotlib; sklearn; warnings; pandas; os; keras; cv2; dlib; pathlib; time






