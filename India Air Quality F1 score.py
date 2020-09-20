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



# With the same split rules, we try different model
# Naive Bayessian
from sklearn.naive_bayes import GaussianNB 


model_bayes = GaussianNB()
model_bayes.fit(X_train,y_train)

#Get the accuracy of the model
model_bayes.score(X_test, y_test) #75.84% accuracy is achieved






#MATRIX FOR EVALUATION 
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
