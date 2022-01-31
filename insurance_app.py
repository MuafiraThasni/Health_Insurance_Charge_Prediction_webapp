# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 23:32:41 2022

@author: muafira
@project: Health Insurance Prediction
"""

#importing required packages
import pandas as pd
import numpy as np

from sklearn import ensemble
import streamlit as st



#Modeling
df_processed = pd.read_csv("processed_data.csv")
X = df_processed.loc[:, df_processed.columns != 'charges']
y = df_processed['charges']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
f_model = ensemble.GradientBoostingRegressor(learning_rate=0.015, n_estimators=250, max_depth=4,min_samples_leaf=5,min_samples_split=10,subsample=1,loss = 'ls',criterion = 'mse')
#f_model.fit(X_train, y_train)
f_model.fit(X, y)
#f_model.predict(X_test)


#print(df_processed.columns)
#print(X.columns)

def predict_insurance(bmi,age,sex,children,smoker,region):
    x = np.zeros(len(X.columns))
    x[1] = bmi
    if sex == 'Female':
        x[2] = 1
    else:
        x[3] = 1
    if children == 0:
        x[4] ==1
    elif children == 1:
        x[5] = 1
    elif children ==2:
        x[6] = 1
    elif children == 3:
        x[7] = 1
    elif children == 4:
        x[8] = 1
    else:
        x[9] = 1
    if age < 40:
        x[18] = 1
    elif age < 58:
        x[16] = 1
    else:
        x[17] = 1
    if smoker == 'Yes':
        x[11] = 1
    else:
        x[10] = 1
    if region == 'North east':
        x[12] = 1
    elif region == 'North west':
        x[13] = 1
    elif region == 'South east':
        x[14] == 1
    else:
        x[15] =1
    return f_model.predict([x])[0]

#testing the prediction with example
print(predict_insurance(28.88, 32, 'Male', 0, 'No', 'North west'))


#web app development
st.title(" ............HealthInsuranceCalc...........")
st.subheader('An AI based prediction for your Health insurance cost....!!!')
st.image("https://cdn.dnaindia.com/sites/default/files/styles/full/public/2021/03/16/964415-health-insurance-istock.jpg")
st.write("""
                  HealthInsuranceCalc is an online tool to predict the health insurance premium cost for 
                  an US resident, given age, bmi, region, sex, number of children and their smoking habit
                  """)
#st.image("https://raw.githubusercontent.com/MuafiraThasni/Health-Insurance-Prediction/main/bmi_age_count.png",width = 700)

st.subheader(" How does the cost depends on the variables...?")
st.write("""Are curious to know the feautures that decides your insurance charge..?
         Data analysis says that your smoking habit, Body Mass Index,
         and age are the most important factors that determine the insurance charges.If you are a
         smoker, your insurance charge probabily will be very high!""")
st.image("https://raw.githubusercontent.com/MuafiraThasni/Health-Insurance-Prediction/main/heatmap.png",width = 300)


#sidebar
st.sidebar.header("Predict Insurance Charge")
region = st.sidebar.selectbox('Region',("North east", "North west","South east","South west"))
bmi = st.sidebar.slider("Your Body-Mass-Index (BMI) ")
age = st.sidebar.slider("How old are you ?")
sex = st.sidebar.selectbox('Sex',("Male", "Female"))
children = st.sidebar.selectbox('Children',(0,1,2,3,4,"5 or more"))
smoker = st.sidebar.selectbox('Smoker',("Yes","No"))

#features = pd.DataFrame({"bmi":bmi,"age":age,"region":region,"sex":sex,"children":children,"smoker":smoker}, index = [0])
predicted_insurance_price =predict_insurance(bmi,age,sex,children,smoker,region)


#display predictions
#st.sidebar.subheader("Prediction")
st.sidebar.write("**Predicted insurance charge is  $**",predicted_insurance_price)
    
