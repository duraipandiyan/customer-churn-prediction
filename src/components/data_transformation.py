import sys 
from dataclasses import dataclass
import numpy as np  
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    Preprocessed_obj_file_path=os.path.join('Artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation=DataTransformationConfig()
        
    def get_data_transformation_obj(self):
        try:
            Numarical_columns=["Total_Orders","Total_Spend","Total_Quantity",
                               "Total_Discount"	,"Avg_Session_Duration"	,"Avg_Pages_Viewed",
                               "Avg_Delivery_Time"	,"Avg_Rating"	]
            
            
    
            Numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('Numerical_pipe', Numerical_pipeline, Numarical_columns)
                ]
            )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transfomation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Successfully loaded train and test data")
            preprocessing_obj = self.get_data_transformation_obj()

            target_column = "Churn"
            remove_column_names = ["Customer_ID", "Recency_Days", "Churn"]

            input_feature_train_data = train_df.drop(columns=remove_column_names, axis=1)
            target_feature_train_data = train_df[target_column]

            input_feature_test_data = test_df.drop(columns=remove_column_names, axis=1)
            target_feature_test_data = test_df[target_column]

            input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train_data)

            input_feature_test_array = preprocessing_obj.transform(input_feature_test_data)

            logging.info("Successfully transformed the data")
            
            train_arr=np.c_[input_feature_train_array, np.array( target_feature_train_data)]
            
            test_arr=np.c_[ input_feature_test_array, np.array(target_feature_test_data)
                           
                           ]
            
            logging.info('successfully preprocessed the object')
            
            
            
            save_object(
                file_path=self.data_transformation.Preprocessed_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr
            )
            
        except Exception as e:
            raise CustomException(e,sys)
        
        