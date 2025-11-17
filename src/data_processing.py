import pandas as pd
import os
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils.helpers import Loader

logger = get_logger(__name__)

class DataProcessing:

    def __init__(self, input_path, output_path):
        
        self.input_path : str = input_path
        self.output_path : str = output_path
        self.df = None

        os.makedirs(self.output_path, exist_ok = True)

        logger.info("Data processing initialized.")

     
    def preprocess(self):
        try:

            logger.info("Data processing started.")

            self.df = Loader.load_data(self.input_path)

            categorical = []
            numerical = []
            for col in self.df.columns:
                if self.df[col].dtype == "object":
                    categorical.append(col)
                else:
                    numerical.append(col)

            self.df["Date"] = pd.to_datetime(self.df["Date"])
            self.df["Year"] = self.df["Date"].dt.year 
            self.df["Month"] = self.df["Date"].dt.month
            self.df["Day"] = self.df["Date"].dt.day

            self.df.drop("Date", axis = 1, inplace = True)

            for col in numerical:
                self.df[col].fillna(self.df[col].mean(), inplace = True)

            self.df.dropna(inplace = True)

            logger.info("Data processing completed successfully.")

        except Exception as e:
            logger.error(f"Error while processing data.")
            raise CustomException(f"Failed to process data.", e)
        
        
    def label_encode(self):

        try:

            categorical = [
                            'Location',
                            'WindGustDir',
                            'WindDir9am',
                            'WindDir3pm',
                            'RainToday',
                            'RainTomorrow'
                          ]
            
            for col in categorical:
                label_encoder = LabelEncoder()
                self.df[col] = label_encoder.fit_transform(self.df[col])
                label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
                logger.info(f"Label Mapping for {self.df[col]} : {label_mapping}")


            logger.info("Label encoding done.")

            
        except Exception as e:
            logger.error(f"Error while label encoding data.")
            raise CustomException(f"Failed to label encode data.", e)
        
    
    def split_data(self):
        try:

            X = self.df.drop("RainTomorrow", axis = 1)
            y = self.df["RainTomorrow"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

            joblib.dump(X_train, X_TRAIN_PATH)
            joblib.dump(X_test, X_TEST_PATH)
            joblib.dump(y_train,y_TRAIN_PATH)
            joblib.dump(y_test, y_TEST_PATH)

            logger.info("Splitted and saved data successfully.")
        

        except Exception as e:
            logger.error(f"Error while splitting and saving data.")
            raise CustomException(f"Failed to split and save data.", e)
        
    
    def run(self):
        try:
            logger.info("Data processing pipeline started.")

            self.preprocess()
            self.label_encode()
            self.split_data()

            logger.info("Data processing pipeline executed successfully.")
        
        except Exception as e:
            logger.error(f"Error while running data processing pipeline.")
            raise CustomException(f"Failed to run data processing pipeline.", e)
        

if __name__ == "__main__":

    processor = DataProcessing(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    processor.run()