# -*- coding: utf-8 -*-


import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier 
import plotly.express as px
import plotly.graph_objects as go
import public_ml_webapp.svmnufhebreast as svmb
# loading the saved models

diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))



# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Encrypted Machine Learning ',
                          
                          ['Flower Classification using EKNN',
                           'Heart Disease Classification using ESVM',
                           'Breast Cancer Classification using ESVM'
                           ],
                          icons=['activity','heart','person'],
                          default_index=0)
    
    
# Diabetes Prediction Page
if (selected == 'Flower Classification using EKNN'):
    # page title
    st.title('Flower Classification using EKNN')
    st.header('User Input Parameters')

    def user_input_features():
        sepal_length = st.slider('Sepal length', 4.3, 7.9, 5.4)
        sepal_width = st.slider('Sepal width', 2.0, 4.4, 3.4)
        petal_length = st.slider('Petal length', 1.0, 6.9, 1.3)
        petal_width = st.slider('Petal width', 0.1, 2.5, 0.2)
        k = st.slider("Choose value of K", min_value=1, max_value=10,key='k')
        data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
        features = pd.DataFrame(data, index=[0])
        return features
    
    df = user_input_features()
    
    st.subheader('User Input parameters')
    st.write(df)
    # Initial Data Plot
    st.dataframe(df)
    fig = px.scatter(df, x = 'sepal_length' , y='sepal_width', symbol='Label',symbol_map={'0':'square-dot' , '1':'circle'})
    fig.add_trace(
        go.Scatter(x= [input[0]], y=[input[1]], name = "Point to Classify", )
    )
    st.plotly_chart(fig)
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    
    clf = RandomForestClassifier()
    clf.fit(X, Y)
    
    prediction = clf.predict(df)
    prediction_proba = clf.predict_proba(df)
    
    st.subheader('Class labels and their corresponding index number')
    st.write(iris.target_names)
    
    st.subheader('Prediction')
    st.write(iris.target_names[prediction])
    #st.write(prediction)
    
    st.subheader('Prediction Probability')
    st.write(prediction_proba)




# Heart Disease Prediction Page
if (selected == 'Heart Disease Classification using ESVM'):
    
    # page title
    st.title('Heart Disease Prediction using ESVM')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction =  ([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
        
    
    

# Parkinson's Prediction Page
if (selected == "Breast Cancer Classification using ESVM"):
    
    # page title
    st.title("Breast Cancer Prediction using ESVM")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Breast cancer Test Result"):
        parkinsons_prediction = svmb.intialization([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB, APQ3]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has cancer"
        else:
          parkinsons_diagnosis = "The person does not have cancer"
        
    st.success(cancer_diagnosis)
















