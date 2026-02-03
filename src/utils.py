import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import numpy as np 
import dill 
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    logging.info('file has been successfully created')
    
def evaluate_model(X_train, Y_train, X_test, Y_test, models, param_grid):
    try:
        report = {}

        for model_name, model in models.items():

            if isinstance(model, LogisticRegression):

                grid = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=5,
                    scoring='f1',
                    n_jobs=-1,
                    verbose=2
                )

                grid.fit(X_train, Y_train)
                best_model = grid.best_estimator_

                y_proba = best_model.predict_proba(X_test)[:, 1]

                thresholds = [0.3, 0.4, 0.5, 0.6]
                for t in thresholds:
                    y_pred_t = (y_proba >= t).astype(int)

                    report[model_name] = {
                        'Accuracy': accuracy_score(Y_test, y_pred_t),
                        'Recall': recall_score(Y_test, y_pred_t),
                        'Precision': precision_score(Y_test, y_pred_t),
                        'F1 Score': f1_score(Y_test, y_pred_t),
                        'ROC-AUC': roc_auc_score(Y_test, y_proba)
                    }
            else:
                model.fit(X_train, Y_train)
                y_proba = model.predict_proba(X_test)[:, 1]

                thresholds = [0.3, 0.4, 0.5, 0.6]
                for t in thresholds:
                    y_pred_t = (y_proba >= t).astype(int)

                    report[model_name] = {
                        'Accuracy': accuracy_score(Y_test, y_pred_t),
                        'Recall': recall_score(Y_test, y_pred_t),
                        'Precision': precision_score(Y_test, y_pred_t),
                        'F1 Score': f1_score(Y_test, y_pred_t),
                        'ROC-AUC': roc_auc_score(Y_test, y_proba)
                    }

        return (report,best_model)

    except Exception as e:
        raise CustomException(e, sys)
    
    
def load_object(file_path):
        try:
            with open(file_path,'rb') as file_obj:
                return dill.load(file_obj)
        except Exception as e:
            raise CustomException(e,sys)
