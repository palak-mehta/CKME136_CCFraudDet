#Import the necessary libraries
from collections import Counter
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn as skl
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import imblearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import plot_roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Load the credit card csv file as a dataframe using pandas
credit_card=pd.read_csv('C:\\Users\\palakm\\creditcardfraud\\creditcard.csv')
print(credit_card)

#Explore the dataset
#See the first 5 rows of the credit_card dataframe
print(credit_card.head())

#See the last 5 rows of the credit_card dataframe
print(credit_card.tail())

#Summarize the shape of the credit_card dataframe
print(credit_card.shape)

#Compute the statistical summary of the credit_card dataframe
print(credit_card.describe())

#Check the missing values in credit_card dataframe
print(credit_card.info())
#Result-Here we can see that there are 284,807 rows and 31 columns in the credit_card dataframe.

print(credit_card.isnull().sum())
# Result-There are no missing values in the dataframe.

#Check unique values in the target variable- 'Class' of the credit_card dataframe
print(credit_card.Class.unique())
#Result-Unique values are 0-Legitimate Transactions and 1-Fraudulent trsnactions.

#check the count of unique values in class attribute of credit_card dataframe
print(credit_card.Class.value_counts())
#Result- The total number of legitimate transactions are 284315 and total number number of fraudulent transactions are 492.

#Check the credit_card data unbalance with respect to target variable-'Class'
legit_trans=len(credit_card[credit_card.Class==0])
print('Total Number of legitmate transactions :', legit_trans)
fraud_trans=len(credit_card[credit_card.Class==1])
print('Total Number of fraudulent transactions :', fraud_trans)

print('Total percentage of legitimate transactions:', round(legit_trans/(legit_trans+fraud_trans) *100,2))
print('Total percentage of fraudulent transactions:',round(fraud_trans/(legit_trans+fraud_trans)*100,2))

# Result-There is severe skewness in the class distribution, with about 99.83 percent of transactions marked as legitimate
# and about 0.17 percent transactions marked as fraudulent.

#See the distribution of the Class attribute
sns.countplot(x='Class',data=credit_card,color='red')
plt.title('Count of Legitimate vs fraudulent transactions \n (0-Legitimate transactions, 1-Fraudulent transactions)')
plt.show()
#Result-The data is highly imbalanced. Most of the transactions are legitimate.

#Heatmap Corrlelation Matrix applied on the credit_card dataset to see the correlation between our predictor variables
# with regards to our target variable 'Class'
cor=credit_card.corr()
fig=plt.figure(figsize=(12,10))
sns.heatmap(cor,square=True,vmax=.8,cmap='PuRd')
plt.title('Imbalanced Data Correlation Matrix')
plt.show()

#Result-There is no noteable coorelation between the features with regards to the tearget variable-'Class.This can be
# probably due to huge class imbalance.

#See the distribution of all the features

creditdata=credit_card.columns.values
i=0
legit_loc=credit_card.loc[credit_card.Class==0]
fraud_loc=credit_card.loc[credit_card.Class==1]

sns.set_style('whitegrid')
plt.figure()
fig.ax=plt.subplots(4,8,figsize=(35,20))

for attributes in creditdata:
    i+=1
    plt.subplot(4,8,i)
    sns.kdeplot(legit_loc[attributes],bw=0.5,label='Legitimate')
    sns.kdeplot(fraud_loc[attributes],bw=0.5,label='Fraudulent')
    plt.xlabel(attributes,fontsize=11)
    locs,labels=plt.xticks()
    plt.tick_params(axis='both',which='major')
plt.show()

#Outliers in this credit_card dataframe are the fraudulent transactions only that deviate from the behaviour of normal
# transaction. IQR was calculated for every feature of the dataframe. Outliers in this case were thethe observations
# that were below (Q1 − 1.5x IQR) or above (Q3 + 1.5x IQR). After removing the outliers there were no fraudulent
# transactions so it was found that Outliers in this credit_card dataframe are frauds only. Therefore not outliers
# were removed.

##Except amount and time features rest of the V1-V28 features are PCA applied(already scaled) so amount and time
# features have been focussed.

##Visualizing of time feature
plt.figure(figsize=(10,6))
sns.distplot(credit_card.Time,color='blue')
plt.title('Distribution of Time feaure')
plt.show()

#Result-The distribution is bimodal. The time is recorded in seconds since the first transaction in the data set.
# Therefore, it can be concluded that this data frame includes all transactions recorded over the course of two days.

#Visualizing of amount feature
plt.figure(figsize=(10,6))
plt.title('Distribution of Amount feaure')
sns.distplot(credit_card.Amount,color='green')
plt.show()

#Result-The figure shows that distribution of the amount of all the transactions is heavily-right skewed. The most of transactions are relatively small and a very small fraction of transactions are close to the maximum.

#In order to apply a PCA transformation all the features need to be scaled and in this dataset all the attributes except time and amount have been previously scaled. So, amount and time features have been scaled.

#Scaling the Amount feature
scaler=RobustScaler()
scaled_Amount=scaler.fit_transform(credit_card['Amount'].values.reshape(-1,1))
print(" Scaled Amount", scaled_Amount)

#Scaling the time feature
scaler=RobustScaler()
scaled_Time=scaler.fit_transform(credit_card['Time'].values.reshape(-1,1))
print("Scaled Time:", scaled_Time)

#Dropping the original Amount and Time feature from the credit_card dataframe
credit_card.drop(['Amount', 'Time'],axis=1,inplace=True)
#Insert scaled amount feature in the credit_card dataframe
credit_card.insert(0,'scaled_Amount',scaled_Amount)

#Insert scaled time feature in the dataframe
credit_card.insert(1,'scaled_Time',scaled_Time)

print("Updated dataset:\n", credit_card.head())

#Resampling techniques to balance the Imbalanced Data
#1. Random Under-Sampling
legit_trans=len(credit_card[credit_card.Class==0])
legit_trans_indices=credit_card[credit_card.Class==0].index
fraud_trans=len(credit_card[credit_card.Class==1])
fraud_trans_indices=credit_card[credit_card.Class==1].index
random_legal_trans_indices=np.random.choice(legit_trans_indices,fraud_trans,replace=False)
under_sample_indices=np.concatenate([fraud_trans_indices,random_legal_trans_indices])
under_sample=credit_card.loc[under_sample_indices]
under_sample_counts=pd.value_counts(under_sample['Class'])
print('Total number of transactions in undersampled credit card dataset are:',len(under_sample))
print('Equal distribution of legitimate and fraudulent transactions in undersampled data:',under_sample_counts)

sns.countplot(x='Class',data=under_sample)
plt.title('Undersampled Data\nEqually Distributed Legitimate and Fraudulent Transactions')
plt.show()

# Now split the under-sampled credit card data into training and test set
x_under=under_sample.drop(['Class'],axis=1)
y_under=under_sample.Class
random_state=7
x_train_under,x_test_under,y_train_under,y_test_under=train_test_split(x_under,y_under,train_size=.70,test_size=.30,random_state=random_state)
print('Total Numer of Transactions in training undersampled dataset:',len(x_train_under))
print('Total Number of Transactions in the testing undersampled dataset:',len(x_test_under))
print('Total number of undersample transactions:',len(x_train_under)+len(x_test_under))


#Model-1. Random Forest Classifier on Undersampled dataframe before feature selection

model_rf=RandomForestClassifier(random_state=random_state)

#Fit a Random Forest Classifier to our training data
model_rf.fit(x_train_under,y_train_under)

#Compute model predictions using testing data
#Actual class predictions
y_under_predict=model_rf.predict(x_test_under)

#Predict probabilities for each class
probs = model_rf.predict_proba(x_test_under)

#Plot the roc curve
rfc_disp = plot_roc_curve(model_rf, x_test_under, y_test_under, alpha=0.8)
plt.title('ROC Curve-RFC Undersampled Dataset')
plt.show()

#Claculate the ROC AUC
print("ROC AUC score: ", round(roc_auc_score(y_test_under, probs[:,1])*100,2),'%')

#Check the accurcacy using actual and predicted values
acc_score=accuracy_score(y_test_under, y_under_predict)
print(" Accuracy score before feature selection is:", round(acc_score*100,2),'%')

#Print the classification report and confusion matrix
print('Classification report:\n',classification_report(y_test_under,y_under_predict))


#Confusion matrix summarizes the information about actual and predicted classifications performed by a classifier
rfc_cfm=confusion_matrix( y_test_under, y_under_predict)
sns.heatmap(rfc_cfm,xticklabels=['Legitimate','Frau='],yticklabels=['Legitimate','Fraud'],annot=True,cmap='Blues',fmt='d')
plt.title("Random Forest Undersampled Data \n Confusion Matrix", fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


#Finding Important Features in the undersampled dataframe using Random Forest Classifier
feature_imp_us=pd.Series(model_rf.feature_importances_,index=x_train_under.columns).sort_values(ascending=False)
print("Important Features:\n", feature_imp_us)

#Visualizing all the features as per their importance score
sns.barplot(x=feature_imp_us,y=feature_imp_us.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Features as per the Importance Score")
plt.show()

#Create a selector object that will use the SelectFromModel to identify features whose importance is greater than the mean importance of all the features
sfm=SelectFromModel(model_rf)
sfm.fit(x_train_under,y_train_under)
selected_feature=x_train_under.columns[(sfm.get_support())]
print(selected_feature)
print('The length of the selected features in the undersampled dataframe using RFC is:',len(selected_feature))

#Create a data subset with only the most important features
x_imptrain_under=sfm.transform(x_train_under)
x_imptest_under=sfm.transform(x_test_under)

#Train Random Forest Classifier using the undersampled feature selected dataset
model_rf_new=RandomForestClassifier(random_state=random_state,n_estimators=100)
model_rf_new.fit(x_imptrain_under,y_train_under)

#Compute model predictions using testing data of undersampled feature selected dataset
#Actual class predictions
y_newunder_predict=model_rf_new.predict(x_imptest_under)

#Predict probabilities for each class
probs_imp = model_rf_new.predict_proba(x_imptest_under)

#Plot the roc curve after applying Random Forest Classifier on the undersampled feature selected dataset
rfc_disp_new = plot_roc_curve(model_rf_new, x_imptest_under, y_test_under, alpha=0.8)
plt.title('ROC Curve-Random Forest Undersampled Feature Selected Dataset')
plt.show()

#Claculate the ROC AUC score after applying Random Forest Classifier on the undersampled feature selected dataset
score_under_rfc=roc_auc_score(y_test_under, probs_imp[:,1])
print("ROC AUC score: ", round(score_under_rfc*100,2),'%')

#Check the accurcacy after applying Random Forest Classifier on the undersampled feature selected dataset
acc_under_rfc=accuracy_score(y_test_under, y_newunder_predict)
print('The accurcay of the dataset after applying RFC on selected features only is:',round(acc_under_rfc*100,2),'%')

#Print the classification report after applying Random Forest Classifier on the undersampled feature selected dataset
print('Classification report:\n',classification_report(y_test_under,y_newunder_predict))

#Confusion matrix after applying Random Forest Classifier on the undersampled feature selected dataset
rfc_cfm_imp=confusion_matrix(y_test_under, y_newunder_predict)
sns.heatmap(rfc_cfm_imp,xticklabels=['Legitimate','Fraud'],yticklabels=['Legitimate','Fraud'],annot=True,cmap='Blues',fmt='d')
plt.title("Random Forest Undersampled Features Selected \n Confusion Matrix", fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#Evaluate Random Forest Classifier Performance on Under sampled feature selected dataset using Cross Validation

#10-Fold Cross Validation

kfold=KFold(n_splits=10,random_state=random_state,shuffle=True)
rfc_score_10=np.mean(cross_val_score(model_rf_new,x_imptrain_under,y_train_under,cv=kfold))
print('The mean accuracy for RFC model using 10 Fold cross validation is:',round(rfc_score_10*100,2),'%')

#5-Fold Cross Validation

kfold=KFold(n_splits=5,random_state=random_state,shuffle=True)
rfc_score_5=np.mean(cross_val_score(model_rf_new,x_imptrain_under,y_train_under,cv=kfold))
print('The mean accuracy for RFC model using 5 Fold cross validation is:',round(rfc_score_5*100,2),'%')

#Model 2. Logistic Regression Classifier-Undersampled feature selected dataset

#Data subset with only the most important features
#x_imptrain_under=sfm.transform(x_train_under)
#x_imptest_under=sfm.transform(x_test_under)

model_lg=LogisticRegression(random_state=random_state)
model_lg.fit(x_imptrain_under,y_train_under)

#Obtain model predictions
y_lg_predict=model_lg.predict(x_imptest_under)

#Predict probabilities for each class
probs_lg = model_lg.predict_proba(x_imptest_under)

#Plot the roc curve
lg_disp_under = plot_roc_curve(model_lg, x_imptest_under, y_test_under, alpha=0.8)
plt.title('ROC Curve-Logistic Regression Undersampled Feature Selected Dataset')
plt.show()

#Claculate the ROC AUC score after applying Logistic Regression Classifier on the undersampled feature selected dataset
score_under_lg=roc_auc_score(y_test_under, probs_lg[:,1])
print("ROC AUC score: ", round(score_under_lg*100,2),'%')

#Check the accurcacy after applying Logistic Regression Classifier on the undersampled feature selected dataset
acc_under_lg=accuracy_score(y_test_under, y_lg_predict)
print('The accurcay of the dataset after applying Logistic Regression Classifier is:',round(acc_under_lg*100,2),'%')

#Print the classification report for the Logistic Regression Classifier
print('Classification report for Logistic Regression Classifier:\n',classification_report(y_test_under,y_lg_predict))

#Confusion matrix for the Logistic Regression Classifier
lg_cfm=confusion_matrix(y_test_under, y_lg_predict)
sns.heatmap(lg_cfm,xticklabels=['Legitimate','Fraud'],yticklabels=['Legitimate','Fraud'],annot=True,cmap='Oranges',fmt='d')
plt.title("Logistic Regression Undersampled Features Selected \n Confusion Matrix", fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#Evaluate Logistic Regression Performance on Under sampled feature selected dataset using Cross Validation

#10-Fold Cross Validation

kfold=KFold(n_splits=10,random_state=random_state,shuffle=True)
lg_score_10=np.mean(cross_val_score(model_lg,x_imptrain_under,y_train_under,cv=kfold))
print('The mean accuracy for LR model using 10 Fold cross validation is:',round(lg_score_10*100,2),'%')

#5-Fold Cross Validation

kfold=KFold(n_splits=5,random_state=random_state,shuffle=True)
lg_score_5=np.mean(cross_val_score(model_lg,x_imptrain_under,y_train_under,cv=kfold))
print('The mean accuracy for LR model using 5 Fold cross validation is:',round(lg_score_5*100,2),'%')

#Model 3. K Nearest Neigbor Classifier-Undersampled feature selected dataset

#Data subset with only the most important features
x_imptrain_under=sfm.transform(x_train_under)
x_imptest_under=sfm.transform(x_test_under)

model_knn=KNeighborsClassifier()
model_knn.fit(x_imptrain_under,y_train_under)

#Obtain model predictions
y_knn_predict=model_knn.predict(x_imptest_under)

#Predict probabilities for each class
probs_knn = model_knn.predict_proba(x_imptest_under)

#Plot the roc curve
knn_disp_under = plot_roc_curve(model_knn, x_imptest_under, y_test_under, alpha=0.8)
plt.title('ROC Curve-KNN Undersampled Feature Selected Dataset')
plt.show()

#Claculate the ROC AUC score after applying KNN Classifier on the undersampled feature selected dataset
score_under_knn=roc_auc_score(y_test_under, probs_knn[:,1])
print("ROC AUC score: ", round(score_under_knn*100,2),'%')

#Check the accurcacy after applying KNN Classifier on the undersampled feature selected dataset
acc_under_knn=accuracy_score(y_test_under,y_knn_predict)
print('The accurcay of the dataset after applying KNN Classifier is:',round(acc_under_knn*100,2),'%')

#Print the classification report for the KNN Classifier
print('Classification report for KNN Classifier:\n', classification_report(y_test_under,y_knn_predict))

#Confusion matrix for the KNN Classifier
knn_cfm=confusion_matrix(y_test_under, y_knn_predict)
sns.heatmap(knn_cfm,xticklabels=['Legitimate','Fraud'],yticklabels=['Legitimate','Fraud'],annot=True,cmap='Greens',fmt='d')
plt.title("KNN Classifier Undersampled Features Selected \n Confusion Matrix", fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#Evaluate KNN Performance on Under sampled feature selected dataset using Cross Validation

#10-Fold Cross Validation

kfold=KFold(n_splits=10,random_state=random_state,shuffle=True)
knn_score_10=np.mean(cross_val_score(model_knn,x_imptrain_under,y_train_under,cv=kfold))
print('The mean accuracy for KNN model using 10 Fold cross validation is:',round(knn_score_10*100,2),'%')

#5-Fold Cross Validation

kfold=KFold(n_splits=5,random_state=random_state,shuffle=True)
knn_score_5=np.mean(cross_val_score(model_knn,x_imptrain_under,y_train_under,cv=kfold))
print('The mean accuracy for KNN model using 5 Fold cross validation is:',round(knn_score_5*100,2),'%')

#2.Random Over-Sampling

#Split the original dataset into training and test set
x=credit_card.drop(['Class'],axis=1)
y=credit_card.Class
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=.70,test_size=.30,random_state=random_state)
print('Total number of Transactions in original data training set:',len(x_train))
print('Total number of Transactions in the original data testing set:',len(x_test))
print('Total number of transactions in original dataset:',len(x_train)+len(x_test))

#Concatenate our training data
z=pd.concat([x_train,y_train],axis=1)

#Apply the over-sampling technique to balance the unbalanced data
ros_legit_data=z[z.Class==0]
ros_fraud_data=z[z.Class==1]
ros_fraud_sample=ros_fraud_data.sample(len(ros_legit_data),replace=True)
ros_data=pd.concat([ros_fraud_sample,ros_legit_data],axis=0)
ros_data_counts=pd.value_counts(ros_data['Class'])
print('Equal distribution of legitimate and fraudulent transactions in oversampled data:',ros_data_counts)
sns.countplot(x='Class',data=ros_data)
plt.title('Oversampled Data\nEqually Distributed Legitimate and Fraudulent Transactions')
plt.show()

ros_y_train=ros_data.Class
ros_x_train=ros_data.drop(['Class'],axis=1)

#Model-1. Random Forest Classifier on Oversampled dataframe before feature selection

over_rf=RandomForestClassifier(random_state=random_state)

#Fit a Random Forest Classifier to our training data
over_rf.fit(ros_x_train,ros_y_train)

#Compute model predictions using testing data
#Actual class predictions
y_over_predict=over_rf.predict(x_test)

#Predict probabilities for each class
probs_over = over_rf.predict_proba(x_test)

#Plot the roc curve
rfc_disp_over = plot_roc_curve(over_rf, x_test, y_test, alpha=0.8)
plt.title('ROC Curve-Random Forest Oversampled Dataset')
plt.show()

#Claculate the ROC AUC
print("ROC AUC score: ", round(roc_auc_score(y_test, probs_over[:,1]),2)*100,'%')

#Check the accurcacy using actual and predicted values
print('The accuracy score before feature selection is:',round(accuracy_score(y_test, y_over_predict)*100,2),'%')


#Print the classification report and confusion matrix
print('Classification report:\n', classification_report(y_test, y_over_predict))

#Confusion matrix summarizes the information about actual and predicted classifications performed by a classifier
rfc_cfm_over=confusion_matrix(y_test, y_over_predict)
sns.heatmap(rfc_cfm_over,xticklabels=['Legitimate','Fraud'],yticklabels=['Legitimate','Fraud'],annot=True,cmap='Blues',fmt='d')
plt.title("Random Forest Oversampled Data \n Confusion Matrix", fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#Finding Important Features in the oversampled dataframe using Random Forest Classifier
feature_imp_os=pd.Series(over_rf.feature_importances_,index=ros_x_train.columns).sort_values(ascending=False)
print("Important Features:\n", feature_imp_os)


#Visualizing all the features as per their importance score
sns.barplot(x=feature_imp_os,y=feature_imp_os.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Features as per the Importance Score")
plt.show()

#Create a selector object that will use the SelectFromModel to identify features whose importance is greater than the mean importance of all the features
sfm_over=SelectFromModel(over_rf)
sfm_over.fit(ros_x_train,ros_y_train)
selected_feature_over=ros_x_train.columns[(sfm_over.get_support())]
print(selected_feature_over)
print('The length of the selected features in the oversampled dataframe after applying RFC is:',len(selected_feature_over))

#Create a data subset with only the most important features
x_imptrain_over=sfm_over.transform(ros_x_train)
x_imptest_over=sfm_over.transform(x_test)

#Train Random Forest Classifier using the oversampled feature selected dataset
over_rf_new=RandomForestClassifier(random_state=random_state)
over_rf_new.fit(x_imptrain_over,ros_y_train)

#Compute model predictions using testing data of oversampled feature selected dataset
#Actual class predictions
y_newover_predict=over_rf_new.predict(x_imptest_over)

#Predict probabilities for each class
probs_over_new = over_rf_new.predict_proba(x_imptest_over)

#Plot the roc curve
rfc_disp_over_new = plot_roc_curve(over_rf_new, x_imptest_over, y_test, alpha=0.8)
plt.title('ROC Curve-Random Forest Oversampled Feature Selected Dataset')
plt.show()

#Claculate the ROC AUC score after applying Random Forest Classifier on the oversampled feature selected dataset
score_over_rfc=roc_auc_score(y_test, probs_over_new[:,1])
print("ROC AUC score: ", round(score_over_rfc*100,2),'%')

#Check the accurcacy after applying Random Forest Classifier on the oversampled feature selected dataset
acc_over_rfc=accuracy_score(y_test, y_newover_predict)
print('The accurcay of the dataset after applying RFC on selected features only is:',round(acc_over_rfc*100,2),'%')

#Print the classification report after applying Random Forest Classifier on the oversampled feature selected dataset
print('Classification report:\n', classification_report(y_test,y_newover_predict))

#Confusion matrix after applying Random Forest Classifier on the oversampled feature selected dataset
rfc_cfm_over_new=confusion_matrix(y_test, y_newover_predict)
sns.heatmap(rfc_cfm_over_new,xticklabels=['Legitimate','Fraud'],yticklabels=['Legitimate','Fraud'],annot=True,cmap='Blues',fmt='d')
plt.title("Random Forest Oversampled Feature Selected \n Confusion Matrix", fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#Evaluate Random Forest Classifier Performance on Over sampled feature selected dataset using Cross Validation

#10-Fold Cross Validation

kfold=KFold(n_splits=10,random_state=random_state,shuffle=True)
rfc_score_over_10=np.mean(cross_val_score(over_rf_new,x_imptrain_over,ros_y_train,cv=kfold))
print('The mean accuracy for RFC model using 10 Fold cross validation is:',round(rfc_score_over_10*100,2),'%')

#5-Fold Cross Validation

kfold=KFold(n_splits=5,random_state=random_state,shuffle=True)
rfc_score_over_5=np.mean(cross_val_score(over_rf_new,x_imptrain_over,ros_y_train,cv=kfold))
print('The mean accuracy for RFC model using 5 Fold cross validation is:',round(rfc_score_over_5*100,2),'%')

#Model 2. Logistic Regression Classifier-Oversampled feature selected dataset


over_lg=LogisticRegression(random_state=random_state)
over_lg.fit(x_imptrain_over,ros_y_train)

#Obtain model predictions
y_over_lg_predict=over_lg.predict(x_imptest_over)

#Predict probabilities for each class
probs_over_lg = over_lg.predict_proba(x_imptest_over)

#Plot the roc curve
lg_disp_over = plot_roc_curve(over_lg, x_imptest_over, y_test, alpha=0.8)
plt.title('ROC Curve-Logistic Regression Oversampled Feature Selected Dataset')
plt.show()

#Claculate the ROC AUC score after applying Logistic Regression Classifier on the oversampled feature selected dataset
score_over_lg=roc_auc_score(y_test, probs_over_lg[:,1])
print("ROC AUC score: ", round(score_over_lg*100,2),'%')

#Check the accurcacy after applying Logistic Regression Classifier on the oversampled feature selected dataset
acc_over_lg=accuracy_score(y_test, y_over_lg_predict)
print('The accurcay of the dataset after applying Logistic Regression Classifier is:',round(acc_over_lg*100,2),'%')

#Print the classification report for the Logistic Regression Classifier on the oversampled feature selected dataset
print('Classification report :\n', classification_report(y_test,y_over_lg_predict))

#Confusion matrix for the Logistic Regression Classifier
lg_cfm_over=confusion_matrix(y_test, y_over_lg_predict)
sns.heatmap(lg_cfm_over,xticklabels=['Legitimate','Fraud'],yticklabels=['Legitimate','Fraud'],annot=True,cmap='Oranges',fmt='d')
plt.title("Logistic Regression Oversampled Features Selected\n Confusion Matrix", fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#Evaluate Logistic Regression Performance on Over sampled feature selected dataset using Cross Validation

#10-Fold Cross Validation

kfold=KFold(n_splits=10,random_state=random_state,shuffle=True)
lg_score_over_10=np.mean(cross_val_score(over_lg,x_imptrain_over,ros_y_train,cv=kfold))
print('The mean accuracy for LR model using 10 Fold cross validation is:',round(lg_score_over_10*100,2),'%')

#5-Fold Cross Validation

kfold=KFold(n_splits=5,random_state=random_state,shuffle=True)
lg_score_over_5=np.mean(cross_val_score(over_lg,x_imptrain_over,ros_y_train,cv=kfold))
print('The mean accuracy for LR model using 5 Fold cross validation is:',round(lg_score_over_5*100,2),'%')

#Model 3. K Nearest Neigbor Classifier-Oversampled feature selected dataset


over_knn=KNeighborsClassifier()
over_knn.fit(x_imptrain_over,ros_y_train)

#Obtain model predictions
y_over_knn_predict=over_knn.predict(x_imptest_over)

#Predict probabilities for each class
probs_over_knn = over_knn.predict_proba(x_imptest_over)

#Plot the roc curve
KNN_disp_over = plot_roc_curve(over_knn, x_imptest_over, y_test, alpha=0.8)
plt.title('ROC Curve-KNN Oversampled Feature Selected Dataset')
plt.show()

#Claculate the ROC AUC score after applying KNN Classifier on the oversampled feature selected dataset
score_over_knn=roc_auc_score(y_test, probs_over_knn[:,1])
print("ROC AUC score: ", round(score_over_knn*100,2),'%')

#Check the accurcacy after applying KNN Classifier on the oversampled feature selected dataset
acc_over_knn=accuracy_score(y_test,y_over_knn_predict)
print('The accurcay of the dataset after applying KNN Classifier is:',round(acc_over_knn*100,2),'%')

#Print the classification report for the KNN Classifier
print('Classification report for KNN Classifier:\n', classification_report(y_test,y_over_knn_predict))

#Confusion matrix for the KNN Classifier
knn_cfm_over=confusion_matrix(y_test, y_over_knn_predict)
sns.heatmap(knn_cfm_over,xticklabels=['Legitimate','Fraud'],yticklabels=['Legitimate','Fraud'],annot=True,cmap='Greens',fmt='d')
plt.title("KNN Classifier Oversampled Features Selected \n Confusion Matrix", fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#Evaluate KNN Performance on Over sampled feature selected dataset using Cross Validation

#10-Fold Cross Validation

kfold=KFold(n_splits=10,random_state=random_state,shuffle=True)
knn_score_over_10=np.mean(cross_val_score(over_knn,x_imptrain_over,ros_y_train,cv=kfold))
print('The mean accuracy for KNN model using 10 Fold cross validation is:',round(knn_score_over_10*100,2),'%')

#5-Fold Cross Validation

kfold=KFold(n_splits=5,random_state=random_state,shuffle=True)
knn_score_over_5=np.mean(cross_val_score(over_knn,x_imptrain_over,ros_y_train,cv=kfold))
print('The mean accuracy for KNN model using 5 Fold cross validation is:',round(knn_score_over_5*100,2),'%')

# 3. Apply SMOTE Technique to balance the unbalanced data
x=credit_card.drop(['Class'],axis=1)
y=credit_card.Class
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=.70,test_size=.30,random_state=random_state)
sm=SMOTE(random_state=random_state)
x_train_sm,y_train_sm=sm.fit_sample(x_train,y_train)
print(Counter(y_train_sm))

#Model-1. Random Forest Classifier on SMOTE dataframe before feature selection

sm_rf=RandomForestClassifier(random_state=random_state)

#Fit a Random Forest Classifier to our training data
sm_rf.fit(x_train_sm,y_train_sm)

#Compute model predictions using testing data
#Actual class predictions
y_sm_predict=sm_rf.predict(x_test)

#Predict probabilities for each class
probs_sm = sm_rf.predict_proba(x_test)

#Plot the roc curve
rfc_disp_sm = plot_roc_curve(sm_rf, x_test, y_test, alpha=0.8)
plt.title('ROC Curve-Random Forest SMOTE Dataset')
plt.show()

#Claculate the ROC AUC
print("ROC AUC score: ", round(roc_auc_score(y_test, probs_sm[:,1])*100,2),'%')

#Check the accurcacy using actual and predicted values
print('The accuracy score before feature selection is:',round(accuracy_score(y_test, y_sm_predict)*100,2),'%')


#Print the classification report and confusion matrix
print('Classification report:\n', classification_report(y_test, y_sm_predict))

#Confusion matrix summarizes the information about actual and predicted classifications performed by a classifier
rfc_cfm_sm=confusion_matrix(y_test, y_sm_predict)
sns.heatmap(rfc_cfm_sm,xticklabels=['Legitimate','Fraud'],yticklabels=['Legitimate','Fraud'],annot=True,cmap='Blues',fmt='d')
plt.title("Random Forest SMOTE Data \n Confusion Matrix", fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


#Finding Important Features in the SMOTE dataframe using Random Forest Classifier
feature_imp_sm=pd.Series(sm_rf.feature_importances_,index=x_train_sm.columns).sort_values(ascending=False)

#Visualizing all the features as per their importance score
sns.barplot(x=feature_imp_sm,y=feature_imp_sm.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Features as per the Importance Score")
plt.show()

#Create a selector object that will use the SelectFromModel to identify features whose importance is greater than the mean importance of all the features
sfm_sm=SelectFromModel(sm_rf)
sfm_sm.fit(x_train_sm,y_train_sm)
selected_feature_sm=x_train_sm.columns[(sfm_sm.get_support())]
print(selected_feature_sm)
print('The length of the selected features in the SMOTE dataframe using RFC is:',len(selected_feature_sm))



#Create a data subset with only the most important features
x_imptrain_sm=sfm_sm.transform(x_train_sm)
x_imptest_sm=sfm_sm.transform(x_test)

#Train Random Forest Classifier using the SMOTE feature selected dataset
sm_rf_new=RandomForestClassifier(random_state=random_state)
sm_rf_new.fit(x_imptrain_sm,y_train_sm)

#Compute model predictions using testing data of SMOTE feature selected dataset
#Actual class predictions
y_newsm_predict=sm_rf_new.predict(x_imptest_sm)

#Predict probabilities for each class
probs_sm_new = sm_rf_new.predict_proba(x_imptest_sm)

#Plot the roc curve
rfc_disp_sm_new = plot_roc_curve(sm_rf_new,x_imptest_sm, y_test, alpha=0.8)
plt.title('ROC Curve-Random Forest SMOTE Feature Selected Dataset')
plt.show()

#Claculate the ROC AUC score after applying Random Forest Classifier on the SMOTE feature selected dataset
score_sm_rfc=roc_auc_score(y_test, probs_sm_new[:,1])
print("ROC AUC score: ", round(score_sm_rfc*100,2),'%')

#Check the accurcacy after applying Random Forest Classifier on the SMOTE feature selected dataset
acc_sm_rfc=accuracy_score(y_test, y_newsm_predict)
print('The accurcay of the dataset after applying RFC on selected features only is:',round(acc_sm_rfc*100,2),'%')

#Print the classification report after applying Random Forest Classifier on the SMOTE feature selected dataset
print('Classification report:\n', classification_report(y_test,y_newsm_predict))

#Confusion matrix after applying Random Forest Classifier on the SMOTE feature selected dataset
rfc_cfm_sm_new=confusion_matrix(y_test, y_newsm_predict)
sns.heatmap(rfc_cfm_sm_new,xticklabels=['Legitimate','Fraud'],yticklabels=['Legitimate','Fraud'],annot=True,cmap='Blues',fmt='d')
plt.title("Random Forest SMOTE Feature Selected \n Confusion Matrix", fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#Evaluate Random Forest Classifier Performance on SMOTE feature selected dataset using Cross Validation

#10-Fold Cross Validation

kfold=KFold(n_splits=10,random_state=random_state,shuffle=True)
rfc_score_sm_10=np.mean(cross_val_score(sm_rf_new,x_imptrain_sm,y_train_sm,cv=kfold))
print('The mean accuracy for RFC model using 10 Fold cross validation is:',round(rfc_score_sm_10*100,2),'%')

#5-Fold Cross Validation

kfold=KFold(n_splits=5,random_state=random_state,shuffle=True)
rfc_score_sm_5=np.mean(cross_val_score(sm_rf_new,x_imptrain_sm,y_train_sm,cv=kfold))
print('The mean accuracy for RFC model using 5 Fold cross validation is:',round(rfc_score_sm_5*100,2),'%')



sm_lg=LogisticRegression(random_state=random_state)
sm_lg.fit(x_imptrain_sm,y_train_sm)

#Obtain model predictions
y_sm_lg_predict=sm_lg.predict(x_imptest_sm)

#Predict probabilities for each class
probs_sm_lg = sm_lg.predict_proba(x_imptest_sm)

#Plot the roc curve
lg_disp_sm= plot_roc_curve(sm_lg, x_imptest_sm, y_test, alpha=0.8)
plt.title('ROC Curve-Logistic Regression SMOTE Feature Selected Dataset')
plt.show()

#Claculate the ROC AUC score after applying Logistic Regression Classifier on the SMOTE feature selected dataset
score_sm_lg=roc_auc_score(y_test, probs_sm_lg[:,1])
print(" ROC AUC score: ", round(score_sm_lg*100,2),'%')

#Check the accurcacy after applying Logistic Regression Classifier on the SMOTE feature selected dataset
acc_sm_lg=accuracy_score(y_test, y_sm_lg_predict)
print('The accurcay of the dataset after applying Logistic Regression Classifier is:',round(acc_sm_lg*100,2),'%')

#Print the classification report for the Logistic Regression Classifier on SMOTE feature selected dataset
print('Classification report :\n', classification_report(y_test,y_sm_lg_predict))

#Confusion matrix for the Logistic Regression Classifier
lg_cfm_sm=confusion_matrix(y_test, y_sm_lg_predict)
sns.heatmap(lg_cfm_sm,xticklabels=['Legitimate','Fraud'],yticklabels=['Legitimate','Fraud'],annot=True,cmap='Oranges',fmt='d')
plt.title("Logistic Regression SMOTE Features Selected\n Confusion Matrix", fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#Evaluate Logistic Regression Performance on SMOTE feature selected dataset using Cross Validation

#10-Fold Cross Validation

kfold=KFold(n_splits=10,random_state=random_state,shuffle=True)
lg_score_sm_10=np.mean(cross_val_score(sm_lg,x_imptrain_sm,y_train_sm,cv=kfold))
print('The mean accuracy for LR model using 10 Fold cross validation is:',round(lg_score_sm_10*100,2),'%')

#5-Fold Cross Validation

kfold=KFold(n_splits=5,random_state=random_state,shuffle=True)
lg_score_sm_5=np.mean(cross_val_score(sm_lg,x_imptrain_sm,y_train_sm,cv=kfold))
print('The mean accuracy for LR model using 5 Fold cross validation is:',round(lg_score_sm_5*100,2),'%')

#Model 3. K Nearest Neigbor Classifier-SMOTE feature selected dataset


sm_knn=KNeighborsClassifier()
sm_knn.fit(x_imptrain_sm,y_train_sm)

#Obtain model predictions
y_sm_knn_predict=sm_knn.predict(x_imptest_sm)

#Predict probabilities for each class
probs_sm_knn = sm_knn.predict_proba(x_imptest_sm)

#Plot the roc curve
knn_disp_sm= plot_roc_curve(sm_knn, x_imptest_sm, y_test, alpha=0.8)
plt.title('ROC Curve-KNN SMOTE Feature Selected Dataset')
plt.show()

#Claculate the ROC AUC score after applying KNN Classifier on the SMOTE feature selected dataset
score_sm_knn=roc_auc_score(y_test, probs_sm_knn[:,1])
print("ROC AUC score: ", round(score_sm_knn*100,2),'%')

#Check the accurcacy after applying KNN Classifier on the SMOTE feature selected dataset
acc_sm_knn=accuracy_score(y_test,y_sm_knn_predict)
print('The accurcay of the dataset after applying KNN Classifier is:',round(acc_sm_knn*100,2),'%')

#Print the classification report for the KNN Classifier
print('Classification report for KNN Classifier:\n', classification_report(y_test,y_sm_knn_predict))

#Confusion matrix for the KNN Classifier
knn_cfm_sm=confusion_matrix(y_test, y_sm_knn_predict)
sns.heatmap(knn_cfm_sm,xticklabels=['Legitimate','Fraud'],yticklabels=['Legitimate','Fraud'],annot=True,cmap='Greens',fmt='d')
plt.title("KNN Classifier SMOTE Features Selected \n Confusion Matrix", fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


#Evaluate KNN Performance on SMOTE feature selected dataset using Cross Validation

#10-Fold Cross Validation

kfold=KFold(n_splits=10,random_state=random_state,shuffle=True)
knn_score_sm_10=np.mean(cross_val_score(sm_knn,x_imptrain_sm,y_train_sm,cv=kfold))
print('The mean accuracy for KNN model using 10 Fold cross validation is:',round(knn_score_sm_10*100,2),'%')

#5-Fold Cross Validation

kfold=KFold(n_splits=5,random_state=random_state,shuffle=True)
knn_score_sm_5=np.mean(cross_val_score(sm_knn,x_imptrain_sm,y_train_sm,cv=kfold))
print('The mean accuracy for KNN model using 5 Fold cross validation is:',round(knn_score_sm_5*100,2),'%')

