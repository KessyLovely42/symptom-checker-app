#import operating system modules
import os

#import data manipulation libraries
import pandas as pd
#silent downcasting warning
pd.set_option('future.no_silent_downcasting', True)

import numpy as np 

#Serialization
import pickle


response = "itching, skin rash, nodal skin eruption, dichromic patches"

def create_empty_indicator_data():
    """
    This function creates an empty dataframe from the dataframe used in training the model
    """
    #indicator file path
    indicator_data_path = os.getcwd()+ "/data_store/train_indicator_data.csv"

    #import empty indicator dataframe and drop the disease colummn
    indicator_data = pd.read_csv(indicator_data_path, nrows =0)
    indicator_data = indicator_data.drop(columns=["Unnamed: 0"]) #

    #create a list of more consistent indicator column namnes for easy processing of user response
    #indicator_columns = [columns.strip().replace(" _","_") for columns in indicator_data.columns]

    #make indicator dataframe columns consistent by replacing columns names with the stripped columns
    #indicator_data.columns = indicator_columns

    return indicator_data

def predict(user_response):
    """
    This function receives user input as a string of symptoms separated by commas, splits the strings to extract 
    the symptoms, creates an empty dataframe akin to the train dataframe before making inferences and providing 
    more information regarding the predicted disease
    """
    #split user response using comma, strip white spaces and convert whitespace inbetween characters to an underscore
    #to correlate with the column names.
    symptom_list = user_response.split(",")
    symptom_list = [symptom.strip().replace(" ","_") for symptom in symptom_list]

    #create the empty indicator dataframe
    indicator_data = create_empty_indicator_data()

    #iterate through the indicators in the symptom lists and append 1 to the respective columns in the dataframe
    for indicator in symptom_list:
        if indicator in indicator_data.columns:
            indicator_data.loc[0,indicator] = 1
            indicator_data = indicator_data.fillna(value = 0) #fill all empty values with 0
    
    #load the serialized model 
    model_path = os.getcwd()+ "/model_store/model.pkl"
    with open(model_path,"rb") as model_pkl:
        model = pickle.load(model_pkl)
    
    prediction = model.predict(indicator_data)

    #Get disease description and precaution
    try:
        description_data = pd.read_csv(os.getcwd() + "/data_store/symptom_Description.csv", index_col="Disease")
        precaution_data = pd.read_csv(os.getcwd() + "/data_store/symptom_precaution.csv", index_col="Disease")
    except FileNotFoundError as e:
        print(f"{e}")
    
    response_msg = {"Disease":prediction, "Description": description_data.loc[prediction,:], "Precation": precaution_data.loc[prediction,:] }
    
    print(response_msg)

def main():
    predict(response)

if __name__ == "__main__":
    main()
    

    
