import os 
import sys 
import pandas as pd
import numpy as np  
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation 
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('Artifacts','train.csv')
    test_data_path=os.path.join('Artifacts','test.csv')
    raw_data_path=os.path.join('Artifacts','raw_data.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        try:
            df=pd.read_csv('Notebook/processed/customer_level_data.csv')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            train_data,test_data=train_test_split(df,test_size=0.2,random_state=42)
            
            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info('Successfully data has been saved from artifacts directory')
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
        
        
if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    
    data_transformation_obj=DataTransformation()
    train_arr,test_arry=data_transformation_obj.initiate_data_transfomation( train_data_path,test_data_path)
    
    model_trainer_obj=ModelTrainer()
    model_report=model_trainer_obj.initiate_model_training(train_arr,test_arry)
    print(model_report)