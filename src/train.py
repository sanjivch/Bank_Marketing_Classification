import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier 
import pickle

#Read the dataset
df = pd.read_csv('/home/elite/Documents/DataScience/Projects/Misc - Machine Learning/Models/bank-additional/bank-additional-full.csv',sep=';')


#Convert categorical values to numeric for each categorical feature
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype('category').cat.codes


#Define function to get all the model metrics
def model_metrics(X_test,y_test,y_model,obj):
    
    conf = confusion_matrix(y_test, y_model)
    
    true_positive = conf[0][0]
    false_negative = conf[1][0]
    true_negative = conf[1][1]
    false_positive = conf[0][1]
    
    sensitivity = true_positive/float(true_positive+false_negative)
    specificity = true_negative/float(true_negative+false_positive)
    mcc = (true_positive*true_negative - false_positive*false_negative)/float((true_positive+false_positive)*(true_positive+false_negative)*(false_positive+true_negative)*(true_negative+false_negative))**0.5
     
    y_pred_proba = obj.predict_proba(X_test)[::,1]
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_test,  y_pred_proba)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    
    
    
    print("Classifier:",obj)
    print("----------------------------------------------------------------------------")
    print("Accuracy\t\t: %0.4f" % accuracy_score(y_test, y_model))
    print("Sensitivity\t\t: %0.4f" % sensitivity)
    print("Specificity\t\t: %0.4f" % specificity)
    print("Matthews Corr. Coeff.\t: %0.4f" % mcc)
    print("----------------------------------------------------------------------------")
    print("Confusion Matrix: \n", conf)
    print("----------------------------------------------------------------------------")
    print("Classification Report: \n",classification_report(y_test, y_model))
    print("----------------------------------------------------------------------------")
    
    plt.title('Receiver Operating Characteristic Curve')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label = 'AUC = %0.4f' % roc_auc)
    plt.legend(loc = 'best')
    plt.plot([0, 1], [0, 1],'r--')
    #plt.xlim([0, 1])
    #plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    


#  Predictor, Taget variables
#Define the predictors and the target variable. No column is being dropped from the predictors.
X = df.drop('y', axis=1)
y = df['y']

#Split the data in 70:30 train-test ratio. We will train the model on X-train, y_train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)


#  Logistic Regression
lr = LogisticRegression(random_state=101)

#Fit the model on training set
model_lr = lr.fit(X_train, y_train)

#Predict on the test set 
pred_lr = model_lr.predict(X_test)


model_metrics(X_test,y_test, pred_lr, model_lr)


#Get the importance of each feature 
def feature_imp(obj):
    print(pd.DataFrame(obj.feature_importances_,index = df.drop('y', axis=1).columns, columns=['imp']).sort_values('imp', ascending = False))


# kFold Cross Validation
# Perform a kFold Cross validation on the model to see if the model is overfitting the data. 

#  Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=101)

# Fit the model on training set
model_rf = rf.fit(X_train, y_train)

# Predict the outcome
pred_rf = model_rf.predict(X_test)

#Model Evaluation
model_metrics(X_test,y_test, pred_rf, model_rf)


#  XGBoost Classifier
xgb = XGBClassifier(learning_rate=0.05, colsample_bylevel=1,colsample_bytree=0.8, max_depth=6, max_delta_step=0.9, n_estimators=300, scale_pos_weight=1, reg_lambda=0.1)

# Fit the model on training set
model_xgb = xgb.fit(X_train, y_train)

# Predict the values for the test set
pred_xgb = model_xgb.predict(X_test)

#Model Evaluation
model_metrics(X_test,y_test, pred_xgb, model_xgb)

# Save models
pickle.dump(model_rf, open('models/bank_marketing_classification_model_rf.pkl', 'wb'))
pickle.dump(model_lr, open('models/bank_marketing_classification_model_lr.pkl', 'wb'))
pickle.dump(model_xgb, open('models/bank_marketing_classification_model_xgb.pkl', 'wb'))
