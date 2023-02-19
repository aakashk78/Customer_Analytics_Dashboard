# Customer_Analytics_Dashboard
Customer Lifetime Value analysis using Machine Learning

This dashboard summarizes results for over 66,000 transactions & 25,000 customers of CDNOW using XGBoostRegressor and XGBoostClassifier

1. Predict customer spending in the next 90 days
2. Predict the probability of customer purchases
3. Segment customers that have high probability but haven't purchased recently

The model has an accuracy of 83.35 %

Files:
1. main.py contains data analysis and XGBoost models
2. The prediction results are stored in Prediction_results.csv
3. web_app.py contains code for dashboard developed using Streamlit
