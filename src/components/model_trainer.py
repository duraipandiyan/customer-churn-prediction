import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                             AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix,classification_report
from src.utils import evaluate_model, save_object
@dataclass
class ModelTrainerConfig:
    model_trainer_file_path=os.path.join('artifacts','model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_training(self,train_arr,test_arr):
        try:
            X_train,Y_train,X_text,Y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
                        
            )
            
            models={
                'RandomForestClassifier':GradientBoostingClassifier(),
                'GradientBoostingClassifier':GradientBoostingClassifier(),
                'AdaBoostClassifier':AdaBoostClassifier(),
                'LogisticRegression':LogisticRegression( solver='liblinear',max_iter=1000,random_state=42),
                'KNeighborsClassifier':KNeighborsClassifier(),
                'DecisionTreeClassifier':DecisionTreeClassifier()
            }
            
            param_grid = {
                'C': [0.01, 0.1, 1, 5, 10],
                'penalty': ['l1', 'l2'],
                'class_weight': [None, 'balanced']
            }
            

        
            model_report,best_model=evaluate_model(X_train,Y_train,X_text,Y_test,models,param_grid)
        
            f1_scores=[]
            for model_name,scores in model_report.items():
                f1_scores.append(scores['F1 Score'])
            best_f1score=max(f1_scores)  

                
            if best_f1score<0.6:
                raise CustomException("No best model Found")
            logging.info('Best found model on both training and dataset')
            
            save_object(
                file_path=self.model_trainer_config.model_trainer_file_path,
                obj=best_model
            )
            
            return (best_f1score, best_model)  
                    
        except Exception as e:
            raise CustomException(e,sys)
    