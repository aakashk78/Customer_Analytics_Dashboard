import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Customer Analytics Dashboard")

col1, col2 = st.columns(2)

col1.subheader('Overview')
col1.markdown("This dashboard summarizes results for over 66,000 transactions & 25,000 customers of CDNOW using XGBoostRegressor and XGBoostClassifier")
col1.markdown("- Predict customer spending in the next 90 days")
col1.markdown("- Predict the probability of customer purchases")
col1.markdown("- Segment customers that have high probability but haven't purchased recently")
pred_value = col1.select_slider('Segment customers based on predicted spending($) :',[50, 100, 150, 200, 250])


predictions_df = pd.read_csv('Prediction_results.csv')
predictions_df['Actual_vs_Predicted Diff'] =predictions_df['spend_90_total'] - predictions_df['pred_spend']

col2.subheader("Actual Spend vs Predicted")

data = predictions_df[(predictions_df['pred_spend'] < pred_value + 0.2*pred_value) & (predictions_df['pred_spend'] > pred_value - 0.2*pred_value)]

fig = px.scatter(data, 'frequency', 'pred_prob',color = 'Actual_vs_Predicted Diff')
col2.write(fig)



col1.subheader('Model Summary:')
col1.markdown("- XGBoost Classifier Accuracy: 83.388%")
col1.markdown("- XGBoost Regressor MAE: 10.135")

if st.checkbox("View Prediction Results", False):
    st.subheader("Predictions")
    st.write(predictions_df)