import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew


#Read the dataset
df = pd.read_csv('/home/elite/Documents/DataScience/Projects/Misc - Machine Learning/Models/bank-additional/bank-additional-full.csv',sep=';')


# ### __Model Building__

# We will model the training dataset with these classifiers
# - Logistic Regression
# - Random Forest
# - XGBoost

# #### __Convert the categorical values to numeric values__

# _scikit-learn_ models do not work with categorial variables(String). Hence, converting them to numeric values.  



#Convert categorical values to numeric for each categorical feature
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype('category').cat.codes



#Define function to get all the model metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
def model_metrics(X_test,y_test,y_model,obj):
    
    conf = confusion_matrix(y_test, y_model)
    tp = conf[0][0]
    fn = conf[1][0]
    tn = conf[1][1]
    fp = conf[0][1]
    
    sens = tp/float(tp+fn)
    spec = tn/float(tn+fp)
    mcc = (tp*tn - fp*fn)/float((tp+fp)*(tp+fn)*(fp+tn)*(tn+fn))**0.5
    
        
    
    y_pred_proba = obj.predict_proba(X_test)[::,1]
    fpr, tpr, threshold = roc_curve(y_test,  y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    
    
#     print "Classifier:",obj
#     print "----------------------------------------------------------------------------"
#     print "Accuracy\t\t: %0.4f" % accuracy_score(y_test, y_model)
#     print "Sensitivity\t\t: %0.4f" % sens
#     print "Specificity\t\t: %0.4f" % spec
#     print "Matthews Corr. Coeff.\t: %0.4f" % mcc
#     print "----------------------------------------------------------------------------"
#     print "Confusion Matrix: \n", conf
#     print "----------------------------------------------------------------------------"
    print("Classification Report: \n",classification_report(y_test, y_model))
#     print "----------------------------------------------------------------------------"
    
    plt.title('Receiver Operating Characteristic Curve')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)
    plt.legend(loc = 'best')
    plt.plot([0, 1], [0, 1],'r--')
    #plt.xlim([0, 1])
    #plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    


# #### __Predictor, Taget variables__

# In[7]:


#Define the predictors and the target variable. No column is being dropped from the predictors.
X = df.drop('y', axis=1)
y = df['y']


# #### __Split the data__

# In[8]:


#Split the data in 70:30 train-test ratio. We will train the model on X-train, y_train set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)


# In[9]:


#Check the size of the training data
X_train.shape


# #### __Logistic Regression__

# In[10]:


from sklearn.linear_model import LogisticRegression


# In[11]:


#Define classifier
lr = LogisticRegression(random_state=101)


# In[12]:


#Fit the model on training set
model_lr = lr.fit(X_train, y_train)


# In[13]:


#Predict on the test set 
pred_lr = model_lr.predict(X_test)


# In[15]:


model_metrics(X_test,y_test, pred_lr, model_lr)


# In[17]:


#Get the importance of each feature 
def feature_imp(obj):
    print(pd.DataFrame(obj.feature_importances_,index = df.drop('y', axis=1).columns, columns=['imp']).sort_values('imp', ascending = False))


# In[18]:


#feature_imp(model_rf)


# > As mentioned in the dataset summary, __duration__ has the largest importance

# #### __Oversampling with SMOTE __
# 
#  The target variable is heavily skewed. We will perform SMOTE to oversample the training dataset. 

# In[470]:


# from imblearn.over_sampling import SMOTE


# # In[506]:


# #define the SMOTE object
# sm = SMOTE(random_state=101)


# In[507]:


# #Fit the sample on the training dataset
# X_sm, y_sm = sm.fit_sample(X_train,y_train)


# # In[508]:


# #Check the fitted sample
# X_sm, y_sm


# # In[498]:


# #Size of the training set after SMOTE
# X_sm.shape, y_sm.shape


# # In[497]:


# #Count of subscribers in Train set after SMOTE
# np.count_nonzero(y_sm == 1)


# # #### __Logistic Regression with SMOTE on training dataset__

# # In[476]:


# #Define classifier
# lr_sm = LogisticRegression()


# # In[477]:


# #Fit the model on SMOTE modified training set
# model_lr_sm = lr_sm.fit(X_sm, y_sm)


# # ##### __kFold Cross Validation__

# # Perform a kFold Cross validation on the model to see if the model is overfitting the data. Applying SMOTE can sometimes overfit the model.

# # In[478]:


# from sklearn.model_selection import cross_val_score
# cvs_lr_sm = cross_val_score(model_lr_sm, X_sm, y_sm, cv=5, n_jobs=3).mean()


# # In[479]:


# print "%0.4f" % cvs_lr_sm


# > Validation accuracy is 86.37%

# In[480]:


# #Prediction on the test set
# pred_lr_sm = model_lr_sm.predict(X_test)


# # In[505]:


# #Model Evaluation
# model_metrics(X_test,y_test, pred_lr_sm, model_lr_sm)


# # #### __Random Forest Classifier__

# In[19]:


from sklearn.ensemble import RandomForestClassifier


# In[20]:


#Define the classifier - 100 trees
rf = RandomForestClassifier(n_estimators=100, random_state=101)


# In[21]:


#Fit the model on training set
model_rf = rf.fit(X_train, y_train)


# In[22]:


# Predict the outcome
pred_rf = model_rf.predict(X_test)


# In[23]:


#Model Evaluation
model_metrics(X_test,y_test, pred_rf, model_rf)


# #### __XGBoost Classifier__

# In[24]:


# from xgboost import XGBClassifier 


# # In[25]:


# #MDefine classifier
# xgb = XGBClassifier(learning_rate=0.05, colsample_bylevel=1,colsample_bytree=0.8, max_depth=6, max_delta_step=0.9, n_estimators=300, scale_pos_weight=1, reg_lambda=0.1)


# # In[484]:


# #Fit the model on training set
# model_xgb = xgb.fit(X_train, y_train)


# # In[485]:


# #Predict the values for the test set
# pred_xgb = model_xgb.predict(X_test)


# # In[486]:


# #Model Evaluation
# model_metrics(X_test,y_test, pred_xgb, model_xgb)


# ### __Summary__
# 
# | Classifier | Accuracy | AUC |
# |------|------|------|------|
# | Logistic Regression  | 0.9091| 0.9250|
# | Logistic Regression + SMOTE | 0.8555| 0.9326|
# | Random Forest | 0.9137| 0.9399|
# | XGBoost | 0.9168| 0.9483|
# 
# - Based on the table above we find that both in terms of *accuracy* and *Area Under the Curve (AUC)*, __XGBoost__ model performs well, followed closely by Random Forest.
# - Logistic Regression with SMOTE gives better AUC, however, performs worse when compared to Logistic regression in terms of accuracy. 

# In[ ]:




