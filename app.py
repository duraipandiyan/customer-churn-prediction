from flask import Flask , request, render_template 
import numpy as np 
import pandas as pd
from src.Pipeline.predict_pipeline import CustomData, PredictPipeline
from sklearn.preprocessing import StandardScaler

Application=Flask(__name__)
app=Application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    else:
        data = CustomData(
            Total_Orders=float(request.form.get('total_orders')),
            Total_Spend=float(request.form.get('total_spend')),
            Total_Quantity=float(request.form.get('total_quantity')),
            Total_Discount=float(request.form.get('total_discount')),
            Avg_Session_Duration=float(request.form.get('avg_session_duration')),
            Avg_Pages_Viewed=float(request.form.get('avg_pages_viewed')),
            Avg_Delivery_Time=float(request.form.get('avg_delivery_time')),
            Avg_Rating=float(request.form.get('avg_rating'))
        )

        pred_df = data.get_data_as_data_frame()
        

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predicts(pred_df)
        
        if result==1:
            lable="Customer Will Churn"
        else:
            lable="Customer Will Not Churn"
        
        return render_template('home.html', result= lable)


if __name__=="__main__":
    app.run()