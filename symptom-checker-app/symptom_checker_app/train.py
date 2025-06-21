from data_processing import *

#Serialization
import pickle

# Model 
from sklearn.ensemble import RandomForestClassifier

def preprocess_data(data):
    """ 
    Function to preprocess data for training
    """
   # Split data
    X = data.drop('Disease', axis = 1)
    Y = data['Disease'] 

    return X, Y

def train(X, Y):
    model = RandomForestClassifier(max_depth= None, max_features= 'sqrt', n_estimators= 100)
    model.fit(X, Y)

    #Serialize trained model to a pickle file for future inference
    #model path 
    model_path = os.getcwd()+"/model_store/model.pkl"
    with open(model_path, "wb") as model_pk:
        pickle.dump(model, model_pk)

def main():
    """ 
    This function contains pipepline of the model training process
    """
    #ingest data
    data = data_ingestion()

    #engineer features
    data = engineer_feature(data)

    #clean data
    data = data_cleaning(data)

    #preprocess data
    X, Y = preprocess_data(data)

    #train and dump model to a pickle file
    train(X,Y)

    #save final data frame used in training data 
    data_path = os.getcwd()+"/data_store/train_indicator_data.csv"
    with open(data_path,"wb") as path:
        X.to_csv(path)

#main method
if __name__ == "__main__":
    main()


    