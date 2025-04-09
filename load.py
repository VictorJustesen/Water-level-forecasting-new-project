# load_data.py
import pandas as pd
import numpy as np
import os

df = pd.read_csv("52.75_Vandstand_Minut.csv", 
                        delimiter = ";", 
                        skiprows = 12,
                        encoding = "unicode-escape")

df=df.drop(columns=['ks mrk.'])
df=df.rename(columns={"Dato (DK normaltid)":"time", "Vandstand (m DVR90)":"level"})
df['time'] = pd.to_datetime(df['time'], format='%d-%m-%Y %H:%M')

df.set_index('time', inplace=True)

df = df.resample('D').mean()

df_temp = pd.read_csv("52.75_Vandtemperatur_Minut.csv", 
                      delimiter=";", 
                      skiprows=12,
                      encoding="unicode-escape")
df_temp = df_temp.drop(columns=['ks mrk.'])
df_temp = df_temp.rename(columns={"Dato (DK normaltid)": "time", "Vandtemperatur (C)": "temp"})
df_temp['time'] = pd.to_datetime(df_temp['time'], format='%d-%m-%Y %H:%M')
df_temp.set_index('time', inplace=True)
df_temp = df_temp.resample('D').mean()  

df = df.join(df_temp, how='outer')


#import weather data
directory_path = 'ClimateData'
all_data = None  

for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)
        
        dfw = pd.read_csv(file_path, usecols=['To', 'Value'])
        
        dfw.rename(columns={'To': 'time'}, inplace=True)
        
        dfw['time'] = dfw['time'].str[:10]
        
        dfw['time'] = pd.to_datetime(dfw['time'], format='%Y-%m-%d')
        
        dfw.set_index('time', inplace=True)
        dfw = dfw.resample('D').mean() 
        clean_filename = os.path.splitext(filename)[0].replace('-', '_')
        dfw.columns = [f'{clean_filename}']
        
        if all_data is None:
            all_data = dfw
        else:
            all_data = all_data.join(dfw, how='outer')

df = df.join(all_data, how='outer')
df = df.dropna()
    
#df['hour'] = df.index.hour
#df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['year'] = df.index.year
df['dayofyear'] = df.index.dayofyear
#df['quarter'] = df.index.quarter
#df['dayofmonth'] = df.index.day
df['weekofyear'] = df.index.isocalendar().week 
  
try:

    df.to_csv("masterdata.csv")
    print("sucess" )
    print(df.columns)

except Exception as e:
    print(f"Error saving DataFrame: {e}")