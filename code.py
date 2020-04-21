# --------------
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
# Path variable
df = pd.read_csv(path)
# First 5 columns
df.head(5)
df.drop('Unnamed: 0',1,inplace=True)
# Independent variables
X = df.drop('SeriousDlqin2yrs',1)
# Dependent variable
y = df['SeriousDlqin2yrs']
# Check the value counts
count = df['SeriousDlqin2yrs'].value_counts()
print(count)
# Split the data set into train and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 6)


# --------------
X = df.drop(['SeriousDlqin2yrs','Unnamed: 0'],axis = 1)
# Dependent variable
y = df['SeriousDlqin2yrs']
# Check the value counts
count = df['SeriousDlqin2yrs'].value_counts()
#print(count)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 6)
# save list of all the columns of X in cols
cols = list(X.columns)
# create subplots
print(cols)
figs,axes = plt.subplots(nrows =5, ncols =2,figsize=(20,20))
for i in range(0,5):
    for j in range(0,2):
        col  = cols[ i * 2 + j]
        axes[i,j].set_title(col)
        axes[i,j].scatter(X_train[col],y_train)
        axes[i,j].set_xlabel(col)
        axes[i,j].set_ylabel('SeriousDlqin2yrs')




# --------------
# Check for null values
print(X_train.isnull().sum())
# Filling the missing values for columns in training data set
from sklearn.preprocessing import Imputer
median_imputer = Imputer(strategy='median')
# Filling the missing values for columns in testing data set
train_imputer = median_imputer.fit(X_train[['MonthlyIncome',]])
test_imputer = median_imputer.fit(X_test[['MonthlyIncome']])
train_imputer = median_imputer.fit(X_train[['NumberOfDependents']])
test_imputer = median_imputer.fit(X_test[['NumberOfDependents']])
X_train['NumberOfDependents'] = train_imputer.transform(X_train[['NumberOfDependents']])
X_test['NumberOfDependents'] = test_imputer.transform(X_test[['NumberOfDependents']])
X_train['MonthlyIncome'] = train_imputer.transform(X_train[['MonthlyIncome']])
X_test['MonthlyIncome'] = test_imputer.transform(X_test[['MonthlyIncome']])

# Checking for null values



# --------------
# Correlation matrix for training set
corr = X_train.corr()
import seaborn as sns
# Plot the heatmap of the correlation matrix
sns.heatmap(corr)
# drop the columns which are correlated amongst each other except one
X_train.drop(['NumberOfTime30-59DaysPastDueNotWorse','NumberOfTime60-89DaysPastDueNotWorse'],1,inplace=True)
X_test.drop(['NumberOfTime30-59DaysPastDueNotWorse','NumberOfTime60-89DaysPastDueNotWorse'],1,inplace=True)


# --------------
from sklearn.preprocessing import StandardScaler
# Initialize scaler object
scaler = StandardScaler()

# Fit on training data with columns of interest
X_train = scaler.fit_transform(X_train)

# Transform test data with columns of interest
X_test = scaler.transform(X_test)


# --------------
# Import Logistic regression model and accuracy score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Instantiate the model in a variable in log_reg
log_reg = LogisticRegression()
# Fit the model on training data
log_reg.fit(X_train,y_train)
# Predictions of the training dataset
y_pred = log_reg.predict(X_test)
# accuracy score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)


# --------------
# Import all the models
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score
# Plot the auc-roc curve
score = roc_auc_score(y_pred , y_test)
y_pred_proba = log_reg.predict_proba(X_test)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="Logistic model, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()

# Evaluation parameters for the model

f1 = f1_score(y_test, log_reg.predict(X_test))
precision = precision_score(y_test, log_reg.predict(X_test))
recall = recall_score(y_test, log_reg.predict(X_test))
roc_auc = roc_auc_score(y_test, log_reg.predict(X_test))
print ('Confusion_matrix' + '\n', confusion_matrix(y_test, log_reg.predict(X_test)))
print ('Classification_report' + '\n' + classification_report(y_test,y_pred))


# --------------
# Import SMOTE from imblearn library
from imblearn.over_sampling import SMOTE

# Check value counts of target variable for data imbalance


# Instantiate smote
smote = SMOTE(random_state=9)
# Fit Smote on training set
X_sample,y_sample = smote.fit_sample(X_train,y_train)
# Check for count of class




# --------------
# Fit logistic regresion model on X_sample and y_sample
log_reg.fit(X_sample,y_sample)
# Store the result predicted in y_pred
y_pred = log_reg.predict(X_test)
# Store the auc_roc score
score = roc_auc_score(y_pred,y_test)
# Store the probablity of any class
y_pred_proba = log_reg.predict_proba(X_test)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
# Plot the auc_roc_graph
plt.plot(fpr,tpr,label="Logistic model, auc="+str(auc))
# Print f1_score,Precision_score,recall_score,roc_auc_score and confusion matrix
f1 = f1_score(y_test, log_reg.predict(X_test))
precision = precision_score(y_test, log_reg.predict(X_test))
recall = recall_score(y_test, log_reg.predict(X_test))
roc_auc = roc_auc_score(y_test, log_reg.predict(X_test))
print ('Confusion_matrix' + '\n', confusion_matrix(y_test, log_reg.predict(X_test)))
print ('Classification_report' + '\n' + classification_report(y_test,y_pred))


# --------------
# Import RandomForestClassifier from sklearn library
from sklearn.ensemble import RandomForestClassifier
# Instantiate RandomForrestClassifier to a variable rf.
rf = RandomForestClassifier(random_state = 9)
# Fit the model on training data.
rf.fit(X_sample,y_sample)
# store the predicted values of testing data in variable y_pred.
y_pred = rf.predict(X_test)
# Store the different evaluation values.
f1 = f1_score(y_test, rf.predict(X_test))
precision = precision_score(y_test, rf.predict(X_test))
recall = recall_score(y_test, rf.predict(X_test))
roc_auc = roc_auc_score(y_test, rf.predict(X_test))
print ('Confusion_matrix' + '\n', confusion_matrix(y_test,rf.predict(X_test)))
print ('Classification_report' + '\n' + classification_report(y_test,y_pred))
# Plot the auc_roc graph
y_pred_proba = rf.predict_proba(X_test)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="XG Boost model, auc="+str(auc))
# Store the auc_roc score
score = roc_auc_score(y_pred,y_test)
plt.plot(fpr,tpr,label="XG Boost model, auc="+str(auc))


