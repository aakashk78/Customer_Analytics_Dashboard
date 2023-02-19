##Import Libraries

import pandas as pd
import numpy as np
import plydata.cat_tools as cat
import matplotlib.pyplot as plt
import seaborn as sns

##Data Import
cdnow_df = pd.read_csv('CDNOW_master.txt', sep = '\s+', names = ['customer_id','date','quantity','price'])

print(cdnow_df.head())
print(cdnow_df.describe())

print(cdnow_df.info())

##Data Preprocessing

cdnow_df['date'] = cdnow_df['date'].astype(str)
print(cdnow_df.head())
cdnow_df['date'] = pd.to_datetime(cdnow_df['date'], format = "%Y%m%d")

print(cdnow_df.info())

##EDA
first_purchase = cdnow_df.sort_values(['customer_id','date']).groupby('customer_id')

first_purchase = first_purchase.first()

print(first_purchase)
print(first_purchase['date'].min())
print(first_purchase['date'].max())

#Price variation by Month
cdnow_df.set_index('date').resample('MS')['price'].sum().plot()

#Time Splitting

n_days = 90
max_date = cdnow_df['date'].max()
cutoff = max_date - pd.to_timedelta(n_days, unit = 'd')

print(cutoff)

## Create datasets before and after cutoff data
in_df = cdnow_df[cdnow_df['date']<=cutoff]

out_df = cdnow_df[cdnow_df['date'] > cutoff]


#Create target_df to be used for training
target_df = out_df.drop('quantity', axis =1).groupby('customer_id').sum().rename({'price':'spend_90_total'},axis=1)
print(target_df)

target_df = target_df.assign(spend_90_flag = 1)
print(target_df)

#Feature Engineering : Recent Purchases
recent_purchases = in_df[['customer_id','date']].groupby('customer_id').apply(lambda x: (x['date'].max() - max_date)/pd.to_timedelta(1,'day')).to_frame().set_axis(['recency'], axis = 1)
print(recent_purchases)

#Feature Engineering : Purchase Frequency
purchase_frequency = in_df[['customer_id','date']].groupby('customer_id').count().set_axis(['frequency'], axis = 1)
print(purchase_frequency)

#Feature Engineering : Price
price_features = in_df.groupby('customer_id').aggregate({'price': ['sum','mean']}).set_axis(['price_mean','price_sum'],axis = 1)
print(price_features)

#Merge Features into features_df
features_df = pd.concat([recent_purchases,purchase_frequency,price_features],axis = 1).merge(target_df, left_index = True, right_index = True, how = 'left').fillna(0)
print(features_df)

#Perform Regression to predict customer spending in next 90 days
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV

#Input features
x = features_df[['recency','frequency','price_sum','price_mean']]
y_spend = features_df['spend_90_total']

#Develop XGBRegressor Model
xgb_reg = XGBRegressor(objective="reg:squarederror", random_state = 123)

xgb_reg_model = GridSearchCV(
    estimator = xgb_reg, 
    param_grid = dict( learning_rate = [0.01,0.1,0.3,0.5]),
    scoring = 'neg_mean_absolute_error',refit = True, cv = 5
)

xgb_reg_model.fit(x, y_spend)

print(xgb_reg_model.best_score_)

print(xgb_reg_model.best_params_)

print(xgb_reg_model.best_estimator_)

predictions_reg = xgb_reg_model.predict(x)

# Determine the probability of customer spending in the next 90 days

y_prob = features_df['spend_90_flag']

#Develop XGBClassifier Model
xgb_clf = XGBClassifier(objective ='binary:logistic',random_state = 123)

xgb_clf_model = GridSearchCV(
    estimator = xgb_clf,
    param_grid = dict(learning_rate = [0.01,0.1,0.3,0.5]),
                      scoring = 'roc_auc',
                      cv = 5
)

xgb_clf_model.fit(x, y_prob)

print(xgb_clf_model.best_score_)

print(xgb_clf_model.best_params_)

print(xgb_clf_model.best_estimator_)

predictions_clf = xgb_clf_model.predict_proba(x)

#Create Dataframe of predictions

predictions_df = pd.concat(
    [pd.DataFrame(predictions_reg).set_axis(['pred_spend'],axis = 1),
     pd.DataFrame(predictions_clf)[[1]].set_axis(['pred_prob'],axis = 1),
     features_df.reset_index()
    ],
    axis = 1
)

#Export Prediction results as csv
predictions_df.to_csv('Prediction_results.csv', index = False)


