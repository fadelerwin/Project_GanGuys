# -*- coding: utf-8 -*-
"""
Created on Sun Sep  11 13:35:34 2020

@author: Fadel
"""

#Import the packages
import pandas as pd
import numpy as np
import datetime

#Import the dataset (We are using 'datasets/city_day.csv' )
air_pollution_type_dataset = pd.read_csv("datasets/station_day.csv", sep = ",")
# passenger_journeys_by_week_by_ticket_type_dataset = pd.read_csv("dataset/Passenger_Journeys_By_Week_By_Ticket_Type.csv", sep = ",")

#Initiate missing values formats
missing_value_formats = ["nan","n.a.","?","NA","n/a", "na", "--"]


#Import the dataset
airquality_df = pd.read_csv("datasets/station_day.csv", sep = ",", na_values = missing_value_formats)


print(airquality_df['O3'].head(10))

#Since the date time is in string, better to chane it to date format (-)
#airquality_df['Date_new']=list(map(lambda x: datetime.datetime.strptime(x,'%d %B %Y').strftime('%d/%m/%Y'), airquality_df['Date'])) (-)


# Null Values in percentage
nan_dataset = airquality_df.isnull().mean().sort_values().round(4)*100



# No missing values in DateTime, Date, and Time (-)


#Make float converter function
def make_float(i):
    try: #try - except
        return float(i)
    except:
        return pd.np.nan

# apply make_int function to the entire series using map
        
# airquality_df['O3_1hr'] = airquality_df['O3_1hr'].map(make_float) (-)

print(airquality_df['O3'].head(10))
print(airquality_df['O3'].isnull().head(10))

#What steps after above code?

# 1st. Aslgorithm. Delete, drop all rows with NaN values
# airquality_df.dropna(axis=0,inplace=True)
# print(airquality_df['O3_1hr'].head(10))

#airquality_df['O3_1hr'].fillna(0, inplace=True)
#print(airquality_df['O3_1hr'].head(10))

#airquality_df['O3_1hr'].fillna(method='bfill', inplace=True)
#print(airquality_df['O3_1hr'].head(10))

# #Not using the below code
# airquality_df["O3_1hr"].interpolate(method='linear', direction = 'forward', inplace=True) 
# airquality_df["O3_1hr"].interpolate(method='linear', direction = 'forward', inplace=True) 
# airquality_df["O3_1hr"].interpolate(method='linear', direction = 'forward', inplace=True) 
# airquality_df["O3_1hr"].interpolate(method='linear', direction = 'forward', inplace=True) 
# print(airquality_df['O3_1hr'].head(10))

#airquality_df[3:15].interpolate(method='linear', direction = 'forward', inplace=True) 

for column_name, rows_over_column in airquality_df.iteritems(): 
    if column_name != 'index' and column_name != 'Name' and column_name != 'GPS' and column_name != 'DateTime' and column_name != 'Date' and column_name != 'Time':
#        print(column_name)
        airquality_df[column_name].interpolate(method='linear', direction = 'forward', inplace=True) 

nan_dataset
nan_dataset2 = airquality_df.isnull().mean().sort_values().round(4)*100


# Finish missing data



# Labeling 

#Creating function to define the label level of each particle
def label_with_NO2(value):
    if value >= 2.0:
        return "Red"
    elif 0.2 <= value and value < 2.0:
        return "Yello"
    elif 0.06 <= value and value < 0.2:
        return "Green"
    else:
        return "Blue"

def label_with_CO(value):
    if value >= 50.0:
        return "Red"
    elif 15.0 <= value and value < 50.0:
        return "Yello"
    elif 9.0 <= value and value < 15.0:
        return "Green"
    else:
        return "Blue"

def label_with_O3(value):
    if value >= 0.5:
        return "Red"
    elif 0.15 <= value and value < 0.5:
        return "Yello"
    elif 0.09 <= value and value < 0.15:
        return "Green"
    else:
        return "Blue"

def label_with_PM10(value):
    if value >= 600.0:
        return "Red"
    elif 150.0 <= value and value < 600.0:
        return "Yello"
    elif 80.0 <= value and value < 150.0:
        return "Green"
    else:
        return "Blue"


def label_with_PM2_5(value):
    if value >= 500.0:
        return "Red"
    elif 75.0 <= value and value < 500.0:
        return "Yello"
    elif 35.0 <= value and value < 75.0:
        return "Green"
    else:
        return "Blue"
    
airquality_df.info()
#Object type: CO, NO2, Date, City, AQI_Bucket



# apply make_int function to the entire series using map
# airquality_df['NO2'] = airquality_df['NO2'].map(label_with_NO2)
# airquality_df['CO'] = airquality_df['CO'].map(label_with_CO)
# airquality_df['O3_1hr'] = airquality_df['O3_1hr'].map(label_with_O3)
# airquality_df['O3_4hr'] = airquality_df['O3_4hr'].map(label_with_O3)
# airquality_df['PM10'] = airquality_df['PM10'].map(label_with_PM10)
# airquality_df['PM2.5'] = airquality_df['PM2.5'].map(label_with_PM2_5)


## MODELLING PHASE 1 ##
airquality_df.info()

# Y_train: AQI index


# AQI calculated by above factors
# Labeling AQI to the X_train
# Output is classifying X_test to AQI


# The Y Factor comes from AQI_Bucket on the city_day.csv file

# We can cleaning the data with removing none on AQI_Bucket


# NOTES #
#Problems with AQI_Bucket --> many missing values (720 rows or 62%)

#Removing null values in AQI_AQI
nan_dataset2

airquality_df['AQI_Bucket'].unique()

#After removing rows contain NA in AQI_Bucket, the rows left are 380 rows
airquality_df2 = airquality_df.dropna(axis = 0, subset = ['AQI_Bucket'])

#To make it easier, change the label in AQI_Bucket to number
#'Poor' = 5, 'Very Poor' = 5, 'Severe' = 4, 'Moderate' = 3, 'Satisfactory' = 2,'Good' = 1

#Change the AQI_Bucket to categorical
airquality_df2['AQI_Bucket'] = pd.Categorical(airquality_df2['AQI_Bucket'])
airquality_df2['AQI_Bucket']
#Get the code for each category and create a new column called AQI_Bucket_code
airquality_df2['AQI_Bucket_code'] = airquality_df2['AQI_Bucket'].cat.codes
airquality_df2['AQI_Bucket_code']

airquality_df2.isnull().mean().sort_values().round(4)*100


# =============================================================================
# We are going to compare the models using F1 Score and recall
# =============================================================================


#START DECESION TREE
# Change label/sytring into number, so it can be modelled
# The AQI calculation uses 7 measures: PM2.5, PM10, SO2, NOx, NH3, CO and O3

#Decide independent variables, these will be assigned to 'X_variables'
airquality_df2.info()


X_variables = airquality_df2[['PM2.5','PM10','NH3','SO2','NOx','CO','O3']]
X_variables                                  

Y_variables = airquality_df2[['AQI_Bucket_code']]
Y_variables

#Import decesion tree model and train_test_split from sklearn'
from sklearn import tree
from sklearn.model_selection import train_test_split


#Split the data into 70 for training 30 for testing
X_train, X_test, y_train, y_test = train_test_split(\
    X_variables, Y_variables, test_size=0.30, random_state=42)


print(X_train.shape)
print(X_test.shape)

#Decided the model to use (For Decesion Trees --> model_dt)
model_dt = tree.DecisionTreeClassifier()
model_dt.fit(X_train,y_train)

#Get the accuracy of the decesion tree model
model_dt.score(X_test, y_test)
model_dt.score(X_train, y_train)

#------------------------------------------------
#MATRIX FOR EVALUATION 
#--------------------------------------------------
#For reference in evaluating ML models: https://www.jeremyjordan.me/evaluating-a-machine-learning-model/
#Import from sklearn.metrics
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
from sklearn.metrics import classification_report

#f1_score(y_test, y_pred)
f1_score(y_test, model_dt.predict(X_test), average = 'micro')
f1_score(y_test, model_bayes.predict(X_test),average='micro')

#Confusion matrix(y_test, y_predict)
confusion_matrix(y_test, model_dt.predict(X_test))
print(classification_report(y_test, model_dt.predict(X_test)))

confusion_matrix(y_test,model_bayes.predict(X_test))
print(classification_report(y_test,model_bayes.predict(X_test)))



# =============================================================================
# TUNING UP the Decesion Tree and Bayesian Parameter Model
# =============================================================================
#https://towardsdatascience.com/how-to-tune-a-decision-tree-f03721801680
#This section for tuning the model so it reachs optimum performances

# Tuning Decesion Tree

#Define range of values I should try for maximum depth
#The minimun number for the leaf node? What is the minimun number?
#Change 1 variable will affect others (no definite rules, try and see)

# Previopus by default: criterion: string, optional (default=”gini”):
# splitter: string, optional (default=”best”)
# Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain

# entropy might be a little slower to compute because it requires you to compute a logarithmic function

# So for a tree with few features without any overfitting, I would go with the “best” splitter to be safe so that you get the best possible model architecture.
#---> try random spliiter alright!

#the “random” splitter has some advantages, specifically, since it selects a set of features randomly and splits

# None case, if you don’t specify a depth for the tree, scikit-learn will expand the nodes until all leaves are pure
# There’s a lot of moving parts here, min_samples_split and min_samples_leaf so let’s just take the max_depth in isolation and see what happens to your model when you change it

# So, if your model is overfitting, reducing the number for max_depth is one way to combat overfitting.
#----> decide max_depth

# https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3





#This is the default one
# DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
#                        max_depth=None, max_features=None, max_leaf_nodes=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=1, min_samples_split=2,
#                        min_weight_fraction_leaf=0.0, presort='deprecated',
#                        random_state=None, splitter='best')

#At default, max_depth = None, proabably means the trees grow as big as possible which may results in overfitting

#TUNING DECEISON TREE (MAX_DEPTH)
#For this deceison tree type 2 model, we will set max_depth
# max_depth to control the size of the tree to prevent overfitting.
# default value of parameter 'max_depth' is None, which means training will continue till all leaves are pure or till all leaves contain < min_samples_split

#max_depth is what the name suggests: The maximum depth that you allow the tree to grow to. The deeper you allow, the more complex your model will become.https://stackoverflow.com/questions/49289187/decision-tree-sklearn-depth-of-tree-and-accuracy


# For testing error, it gets less obvious. If you set max_depth too high, then the decision tree might simply overfit the training data without capturing useful patterns as we would like; this will cause testing error to increase. 
# But if you set it too low, that is not good as well; then you might be giving the decision tree too little flexibility to capture the patterns and interactions in the training data. This will also cause the testing error to increase
# Usually, the modeller would consider the max_depth as a hyper-parameter, and use some sort of grid/random search with cross-validation to find a good number for max_depth

 # The best hyperparameters are usually impossible to determine ahead of time, and tuning a model is where machine learning turns from a science into trial-and-error based engineering.https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
#When a model performs highly on the training set but poorly on the test set, this is known as overfitting
# Therefore, the standard procedure for hyperparameter optimization accounts for overfitting through cross validation


#After trying this, using random search grid please
max_depth_set = 3 #recommended by sckiti learn as initial number and start see the tree grows

model_dt_type2 = tree.DecisionTreeClassifier(max_depth=max_depth_set)
model_dt_type2.fit(X_train,y_train)

#Get the accuracy of the decesion tree model
model_dt_type2.score(X_test, y_test)
model_dt_type2.score(X_train, y_train)

#Confusion matrix(y_test, y_predict)
confusion_matrix(y_test, model_dt_type2.predict(X_test))
print(classification_report(y_test, model_dt_type2.predict(X_test),digits = 4))


#Random search grid to find the best max_depth

from sklearn.model_selection import RandomizedSearchCV

params1 = {'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)] }

 #Running random search to find the best max_depth

   #Our default model is model_dt

model_dt

model_dt_max_depth1 = RandomizedSearchCV(estimator = model_dt, param_distributions = params1)
   #After we find the best model in model_dt_max_depth then we fit it
model_dt_max_depth1.fit(X_train, y_train)    
   #This is our best max_depth we could get from random search and trial-error which is 'max_depht' = 10
model_dt_max_depth1.best_params_
    #Get the accuracy,reall,f1-score
confusion_matrix(y_test, model_dt_max_depth1.predict(X_test))
print(classification_report(y_test, model_dt_type2.predict(X_test), digits = 4))



#After that we could change the splitter type into 'random'
model_dt 

#https://stackoverflow.com/questions/46756606/what-does-splitter-attribute-in-sklearns-decisiontreeclassifier-do
# Splitter:
# The splitter is used to decide which feature and which threshold is used.
# Using best, the model if taking the feature with the highest importance
# Using random, the model if taking the feature randomly but with the same distribution (in gini, proline have an importance of 38% so it will be taken in 38% of cases)

splitter_par1 = 'random'

model_dt_splitter1 = tree.DecisionTreeClassifier(splitter = splitter_par1)
model_dt_splitter1.fit(X_train,y_train)

#Confusion matrix(y_test, y_predict)
confusion_matrix(y_test, model_dt_splitter1.predict(X_test))
print(classification_report(y_test, model_dt_splitter1.predict(X_test), digits = 4))


#Next, we will put it on the table and chart for comparison




    
 #BAYESIAN TUNING (CANT BE DONE: NO OPTIMIZATION FOR BAYESIAN )
#This is our previous or default model_bayes
model_bayes

# Hyperparameter tuning by means of Bayesian reasoning, or Bayesian Optimisation, can bring down the time spent to get to the optimal set of parameters — and bring better generalisation performance on the test set
# https://medium.com/vantageai/bringing-back-the-time-spent-on-hyperparameter-tuning-with-bayesian-optimisation-2e21a3198afb
# https://machinelearningmastery.com/scikit-optimize-for-hyperparameter-tuning-in-machine-learning/
# this is from scratch: https://machinelearningmastery.com/what-is-bayesian-optimization/

# Scikit-Optimize, or skopt for short, is an open-source Python library for performing optimization tasks.

#Tuning hyperparameter: GridSearch, RandomSearch (Very common)


# pip instal scikit-optimize

# report scikit-optimize version number
import skopt
print('skopt %s' % skopt.__version__)




# =============================================================================
# SKIP ROC CURVE FOR A MOMENT
# =============================================================================
# #ROC CURVE
# # https://github.com/dataprofessor/code/blob/master/python/ROC_curve.ipynb

# #For building the ROC curves, it needs the value of each prediction done in Decesion Tree and Naive Bayesian
# #We are using base parameters to compare ROC values between Decesion Tree (Default) and Naive Bayesian (Default)


# r_probs = [0 for _ in range(len(y_test))]

# dt_probs = model_dt.predict_proba(X_test)
# bayes_probs = model_bayes.predict_proba(X_test)


# # Probabilities for the positive outcome is kept.
# dt_probs = dt_probs[:, 1]
# bayes_probs = bayes_probs[:, 1]

# class_type = np.unique(airquality_df2['AQI_Bucket_code'])


# false_positive_dt, true_positive_dt, threshold_dt = roc_curve(y_test, dt_probs)




# roc_auc_score(y_test,dt_probs,multi_class = 'ovr')
# roc_auc_score(y_test,bayes_probs,multi_class = 'ovr')



# # Computing AUROC and ROC curve values
# from sklearn.metrics import roc_curve, roc_auc_score
# # Calculate AUROC
# # ROC is the receiver operating characteristic AUROC is the area under the ROC curve

# r_auc = roc_auc_score(y_test, r_probs,multi_class = 'ovr')
# dt_auc = roc_auc_score(y_test,dt_probs,multi_class = 'ovr')
# bayes_auc = roc_auc_score(y_test,bayes_probs,multi_class = 'ovr')


# # Print AUROC scores
# print('Random (chance) Prediction: AUROC = %.3f' % (r_auc))
# print('Random Forest: AUROC = %.3f' % (rf_auc))
# print('Naive Bayes: AUROC = %.3f' % (nb_auc))

# # Calculate ROC curve
# r_fpr, r_tpr, _ = roc_curve(Y_test, r_probs)
# rf_fpr, rf_tpr, _ = roc_curve(Y_test, rf_probs)
# nb_fpr, nb_tpr, _ = roc_curve(Y_test, nb_probs)




# With the same split rules, we try different model
# Naive Bayessian
from sklearn.naive_bayes import GaussianNB 


model_bayes = GaussianNB()
model_bayes.fit(X_train,y_train)

#Get the accuracy of the model
model_bayes.score(X_test, y_test) #75.84% accuracy is achieved






