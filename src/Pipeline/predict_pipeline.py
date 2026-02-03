import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predicts(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features) 
            y_proba=model.predict_proba(data_scaled)[:,1][0]
            y_pred = (y_proba >= 0.4).astype(int)
            
            
            return y_pred
        
        except Exception as e:
            raise CustomException(e,sys)
    
class CustomData():
    def __init__(self,
                 Total_Orders,
                 Total_Spend,
                 Total_Quantity,
                 Total_Discount,
                 Avg_Session_Duration,
                 Avg_Pages_Viewed	,
                 Avg_Delivery_Time,
                 Avg_Rating):
        
        self.Total_Orders= Total_Orders
        self.Total_Spend=Total_Spend
        self.Total_Quantity=Total_Quantity
        self.Total_Discount=Total_Discount
        self.Avg_Session_Duration=Avg_Session_Duration
        self.Avg_Pages_Viewed=Avg_Pages_Viewed
        self.Avg_Delivery_Time=Avg_Delivery_Time
        self.Avg_Rating=Avg_Rating

    def get_data_as_data_frame(self):
        
        try:
            custom_data_input_dic={
                "Total_Orders":[self.Total_Orders],
                "Total_Spend":[self.Total_Orders],
                'Total_Quantity':[self.Total_Quantity],
                'Total_Discount':[self.Total_Discount],
                'Avg_Session_Duration':[self.Avg_Session_Duration],
                "Avg_Pages_Viewed":[self.Avg_Pages_Viewed],
                'Avg_Delivery_Time':[self.Avg_Delivery_Time],
                'Avg_Rating':[self.Avg_Rating]
            }
            
            return pd.DataFrame(custom_data_input_dic)
        
        except Exception as e:
            raise CustomException(e,sys)
        