import pandas as pd
import numpy as np
import datetime

air_pollution_type_dataset = pd.read_csv("dataset/air_pollution.csv", sep = ",")
# passenger_journeys_by_week_by_ticket_type_dataset = pd.read_csv("dataset/Passenger_Journeys_By_Week_By_Ticket_Type.csv", sep = ",")
missing_value_formats = ["nan","n.a.","?","NA","n/a", "na", "--"]
airquality_df = pd.read_csv("dataset/Air_Quality_Monitoring_Data.csv", sep = ",", na_values = missing_value_formats, nrows=1000)

print(airquality_df['O3_1hr'].head(10))

#Since the date time is in string, better to chane it to date format 
#airquality_df['Date_new']=list(map(lambda x: datetime.datetime.strptime(x,'%d %B %Y').strftime('%d/%m/%Y'), airquality_df['Date']))
# Null Values in percentage
nan_dataset = airquality_df.isnull().mean().sort_values().round(4)*100
# No missing values in DateTime, Date, and Time.

def make_float(i):
    try:
        return float(i)
    except:
        return pd.np.nan

# apply make_int function to the entire series using map
# airquality_df['O3_1hr'] = airquality_df['O3_1hr'].map(make_float)

print(airquality_df['O3_1hr'].head(10))
print(airquality_df['O3_1hr'].isnull().head(10))


# 1st. Aslgorithm. Delete, drop all rows with NaN values
# airquality_df.dropna(axis=0,inplace=True)
# print(airquality_df['O3_1hr'].head(10))

#airquality_df['O3_1hr'].fillna(0, inplace=True)
#print(airquality_df['O3_1hr'].head(10))

#airquality_df['O3_1hr'].fillna(method='bfill', inplace=True)
#print(airquality_df['O3_1hr'].head(10))


airquality_df["O3_1hr"].interpolate(method='linear', direction = 'forward', inplace=True) 
airquality_df["O3_1hr"].interpolate(method='linear', direction = 'forward', inplace=True) 
airquality_df["O3_1hr"].interpolate(method='linear', direction = 'forward', inplace=True) 
airquality_df["O3_1hr"].interpolate(method='linear', direction = 'forward', inplace=True) 
print(airquality_df['O3_1hr'].head(10))

#airquality_df[3:15].interpolate(method='linear', direction = 'forward', inplace=True) 

for column_name, rows_over_column in airquality_df.iteritems(): 
    if column_name != 'index' and column_name != 'Name' and column_name != 'GPS' and column_name != 'DateTime' and column_name != 'Date' and column_name != 'Time':
#        print(column_name)
        airquality_df[column_name].interpolate(method='linear', direction = 'forward', inplace=True) 

nan_dataset = airquality_df.isnull().mean().sort_values().round(4)*100


# Finish missing data
# Labeling 


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

# apply make_int function to the entire series using map
airquality_df['NO2'] = airquality_df['NO2'].map(label_with_NO2)
airquality_df['CO'] = airquality_df['CO'].map(label_with_CO)
airquality_df['O3_1hr'] = airquality_df['O3_1hr'].map(label_with_O3)
airquality_df['O3_4hr'] = airquality_df['O3_4hr'].map(label_with_O3)
airquality_df['PM10'] = airquality_df['PM10'].map(label_with_PM10)
airquality_df['PM2.5'] = airquality_df['PM2.5'].map(label_with_PM2_5)



# 


'''
airquality_df["PM10"] = airquality_df["PM10"].interpolate(method='linear', direction = 'forward', inplace=True) 
airquality_df["AQI_Site"] = airquality_df["AQI_Site"].interpolate(method='linear', direction = 'forward', inplace=True) 
airquality_df["AQI_PM10"] = airquality_df["AQI_PM10"].interpolate(method='linear', direction = 'forward', inplace=True) 
airquality_df["O3_1hr"] = airquality_df["O3_1hr"].interpolate(method='linear', direction = 'forward', inplace=True) 
airquality_df["AQI_O3_1hr"] = airquality_df["AQI_O3_1hr"].interpolate(method='linear', direction = 'forward', inplace=True) 
airquality_df["O3_4hr"] = airquality_df["O3_4hr"].interpolate(method='linear', direction = 'forward', inplace=True) 
airquality_df["AQI_O3_4hr"] = airquality_df["AQI_O3_4hr"].interpolate(method='linear', direction = 'forward', inplace=True) 
airquality_df["AQI_PM2.5"] = airquality_df["AQI_PM2.5"].interpolate(method='linear', direction = 'forward', inplace=True) 
airquality_df["PM2.5"] = airquality_df["PM2.5"].interpolate(method='linear', direction = 'forward', inplace=True) 
airquality_df["AQI_NO2"] = airquality_df["AQI_NO2"].interpolate(method='linear', direction = 'forward', inplace=True) 
airquality_df["O3_4hr"] = airquality_df["O3_4hr"].interpolate(method='linear', direction = 'forward', inplace=True) 
airquality_df["AQI_CO"] = airquality_df["AQI_CO"].interpolate(method='linear', direction = 'forward', inplace=True) 
airquality_df["NO2"] = airquality_df["NO2"].interpolate(method='linear', direction = 'forward', inplace=True) 
airquality_df["CO"] = airquality_df["CO"].interpolate(method='linear', direction = 'forward', inplace=True) 
'''