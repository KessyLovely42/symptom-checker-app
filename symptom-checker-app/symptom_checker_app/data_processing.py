#import operating system resources
import os

# fundamental python libraries
import numpy as np
import pandas as pd

# High quality data visualization
import matplotlib.pyplot as plt
import seaborn as sns

def data_ingestion():
    try:
       data = pd.read_csv(os.getcwd() + "/data_store/dataset.csv")
       return data
    except FileNotFoundError as e:
        print("Cannot find dataset in the specified path")
   

def engineer_feature(data):
    # Get all the columns except the target
    cols = data.drop('Disease', axis=1).columns
    # List all unique values in the dataset, 
    # These unique values represents the symptoms to be used as the indicator features
    unique_values = pd.unique(data[cols].values.ravel())
    unique_values = [value for value in unique_values if pd.notna(value)]  # Remove NaN

    # Remove trailing white spaces from the unique values
    data = data.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Update the unique values
    unique_values = [value.strip() for value in unique_values]

    # Create the indicator df with the unique values (symptoms), initialize each cell to False
    indicator_df = pd.DataFrame(False, index=data.index, columns=unique_values)

    # Iterate through each row in the main dataframe
    # Update the value of the features in the indicator df with the occurrence in each rows
    for index, row in data[cols].iterrows():
        for value in row:
            if pd.notna(value):  # Ignore NaN
                indicator_df.loc[index, value] = True
    
    # Join the indicator df with the target from the main dataframe
    data = pd.concat([data['Disease'], indicator_df], axis=1)
    
    return data
      
def data_cleaning(data):
    # Drop duplicate entries
    data = data.drop_duplicates()
    data.reset_index()
    # Drop irrelevant symptoms
    data = data.drop(labels=['extra_marital_contacts', 'history_of_alcohol_consumption', 'receiving_unsterile_injections', 'family_history', 'receiving_blood_transfusion'], axis=1)

    # Filter out rows where all values are False
    no_symptom = data.groupby('Disease').sum().sum(axis=1)
    for key, value in no_symptom.items():
        if value == 0:
            data = data[data['Disease'] != key]
    data.reset_index()

    #create list of column names using underscore as the only separator and assign to column names of dataframe
    data_columns = [columns.strip().replace(" _","_") for columns in data.columns]
    data.columns = data_columns
    return data