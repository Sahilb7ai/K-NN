import os
import pandas as pd

os.chdir("C:/Users/sahil/Documents/IMR/Data_KNN/")

FullRaw = pd.read_csv("cancerdata.csv")

###################
# Missing Value Check
###################

FullRaw.isnull().sum()


###################
# Recode Categorical variables to Numeric (dummy variable creation)
###################

import numpy as np

FullRaw.dtypes # There is only ONE categ. var, i.e. diagnosis (Dependent variable)
FullRaw['diagnosis'] = np.where(FullRaw['diagnosis'] == 'M', 1, 0)

###################
# Drop id column
###################

FullRaw.drop(['id'], axis = 1, inplace = True)
FullRaw.shape

###################
# Sampling
###################

from sklearn.model_selection import train_test_split
Train, Test = train_test_split(FullRaw, train_size = 0.8, random_state = 123)

depVar = 'diagnosis'
trainX = Train.drop([depVar], axis = 1).copy()
trainY = Train[depVar].copy()
testX = Test.drop([depVar], axis = 1).copy()
testY = Test[depVar].copy()

trainX.shape
testX.shape

###################
# Standardization
###################

from sklearn.preprocessing import StandardScaler

Train_Scaling = StandardScaler().fit(trainX) # Train_Scaling contains means, std_dev of training dataset
trainX_Std = Train_Scaling.transform(trainX) # This step standardizes (executes the formula) the train data
testX_Std  = Train_Scaling.transform(testX) # This step standardizes (executes the formula) the test data

# Add the column names to trainX_Std, testX_Std
trainX_Std = pd.DataFrame(trainX_Std, columns = trainX.columns)
testX_Std = pd.DataFrame(testX_Std, columns = testX.columns)

###################
# Model building
###################

from sklearn.neighbors import KNeighborsClassifier
M1 = KNeighborsClassifier(n_neighbors = 3).fit(trainX_Std, trainY)

###################
# Model prediction
###################

# Class Prediction
testPred = M1.predict(testX_Std)

# Create a df to store the model predictions
Test_Prob_Df = pd.DataFrame()
Test_Prob_Df['Class'] = testPred

# Probability Prediction: Probability is just the fraction of 0s and 1s in the K nearest neighbours
Test_Prob = pd.DataFrame(M1.predict_proba(testX_Std))
Test_Prob_Df = pd.concat([Test_Prob_Df, Test_Prob], axis = 1)


###################
# Model evaluation
###################

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

Confusion_Mat = confusion_matrix(testY, testPred)
Confusion_Mat

sum(np.diagonal(Confusion_Mat))/testX.shape[0]*100 # Accuracy
precision_score(testY, testPred) # Precision [Total Positives/(Total Predicted Positives)]
recall_score(testY, testPred) # Recall (Also called TPR) [Total Positives/(Total Actual Positives)]
f1_score(testY, testPred) # F1-Score [2*Precision*Recall/(Precision + Recall)] 
Confusion_Mat[0][1]/sum(Confusion_Mat[0]) # FPR [FP/Total Actual 0s(Negatives)]


###################
# Grid Search CV
###################
             
from sklearn.model_selection import GridSearchCV

myNN = range(1,14,2) # list(range(1,14,2))
myP = range(1,4,1) # list(range(1,4,1)) . This "p" is "h" in minkowski formula
my_param_grid = {'n_neighbors': myNN, 'p': myP} # param_grid is a dictionary 

Grid_Search_Model = GridSearchCV(estimator = KNeighborsClassifier(), 
                     param_grid=my_param_grid,  
                     scoring='accuracy', 
                     cv=5, n_jobs = -1).fit(trainX_Std, trainY)
# Other scoring parameters are available here: http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

Grid_Search_Df = pd.DataFrame.from_dict(Grid_Search_Model.cv_results_)


###################      
###################
# Data Re-Distribution (Changing data when there are outliers present in some columns)
# There is a good belief that some times outliers influence and interfere in the way algorithms work
# So, it may be needed to change the way data is originally distributed and feed the changed data
# into an algorithm.
###################
###################           
           
import seaborn as sns
import numpy as np

trainX.columns
columnsToConsider = ["perimeter_mean", "area_mean", "perimeter_worst", "area_worst"]

# Histogram using seaborn
sns.pairplot(trainX[columnsToConsider])

# Lets consider Area Mean and apply log transformation
trainX_Copy = trainX.copy()
trainX_Copy["area_mean"] = np.log(np.where(trainX_Copy["area_mean"] == 0, 1, trainX_Copy["area_mean"]))
trainX_Copy["perimeter_worst"] = np.log(np.where(trainX_Copy["perimeter_worst"] == 0, 1, trainX_Copy["perimeter_worst"]))
trainX_Copy["area_worst"] = np.log(np.where(trainX_Copy["area_worst"] == 0, 1, trainX_Copy["area_worst"]))

testX_Copy = testX.copy()
testX_Copy["area_mean"] = np.log(np.where(testX_Copy["area_mean"] == 0, 1, testX_Copy["area_mean"]))
testX_Copy["perimeter_worst"] = np.log(np.where(testX_Copy["perimeter_worst"] == 0, 1, testX_Copy["perimeter_worst"]))
testX_Copy["area_worst"] = np.log(np.where(testX_Copy["area_worst"] == 0, 1, testX_Copy["area_worst"]))

# Histogram using seaborn
sns.pairplot(trainX_Copy[columnsToConsider])


###################
# Standardization
###################

Train_Scaling = StandardScaler().fit(trainX_Copy) # Train_Scaling contains means, std_dev of training dataset
trainX_Std = Train_Scaling.transform(trainX_Copy) # This step standardizes the train data
testX_Std  = Train_Scaling.transform(testX_Copy) # This step standardizes the test data

# Add the column names to trainX_Std, testX_Std
trainX_Std = pd.DataFrame(trainX_Std, columns = trainX.columns)
testX_Std = pd.DataFrame(testX_Std, columns = testX.columns)



###################
# Modeling
###################

# Build model
M2 = KNeighborsClassifier(n_neighbors = 3).fit(trainX_Std, trainY)

# Class Prediction
testPred = M2.predict(testX_Std)

###################
# Model evaluation
###################

Confusion_Mat = confusion_matrix(testY, testPred)
Confusion_Mat

sum(np.diagonal(Confusion_Mat))/testX.shape[0]*100 # Accuracy
precision_score(testY, testPred) # Precision [Total Positives/(Total Predicted Positives)]
recall_score(testY, testPred) # Recall (Also called TPR) [Total Positives/(Total Actual Positives)]
f1_score(testY, testPred) # F1-Score [2*Precision*Recall/(Precision + Recall)] 
Confusion_Mat[0][1]/sum(Confusion_Mat[0]) # FPR [FP/Total Actual 0s(Negatives)]

