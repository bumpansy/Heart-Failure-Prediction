import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
#from EDA.EDA import *
import tensorflow as tf
from tensorflow.keras.models import load_model

#model = load_model('E:\Public Repositories\Heart-Failure-Prediction\Model\my_model.h5')

model = tf.keras.saving.load_model(
    'E:\Public Repositories\Heart-Failure-Prediction\Model\my_model.tf', custom_objects=None, compile=True, safe_mode=True
)

st.set_page_config(layout='wide')

url = 'https://raw.githubusercontent.com/bumpansy/Heart-Failure-Prediction/main/Model/heart.csv'
df = pd.read_csv(url)

#model = pickle.load(open('E:\\Public Repositories\\Heart-Failure-Prediction\\Model\\newmodel.pkl', 'rb'))

def user_input_features():
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input('Enter your age: ')
        sex  = st.selectbox('Sex',(0,1))
        cp = st.selectbox('Chest pain type',(0,1,2,3))
        tres = st.number_input('Resting blood pressure: ')
        chol = st.number_input('Serum cholestoral in mg/dl: ')
        fbs = st.selectbox('Fasting blood sugar',(0,1))
        res = st.number_input('Resting electrocardiographic results: ')
    with col2:
        tha = st.number_input('Maximum heart rate achieved: ')
        exa = st.selectbox('Exercise induced angina: ',(0,1))
        old = st.number_input('oldpeak ')
        slope = st.number_input('he slope of the peak exercise ST segmen: ')
        ca = st.selectbox('number of major vessels',(0,1,2,3))
        thal = st.selectbox('thal',(0,1,2))

    data = {'age': age,
            'sex': sex, 
            'cp': cp,
            'trestbps':tres,
            'chol': chol,
            'fbs': fbs,
            'restecg': res,
            'thalach':tha,
            'exang':exa,
            'oldpeak':old,
            'slope':slope,
            'ca':ca,
            'thal':thal
                }
    features = pd.DataFrame(data, index=[0])
    return features

st.header('Heart Failure Prediction App')
st.write('This app is made to predict the probability of heart failure in a person using his/her vital information.')

tab1,tab2,tab3,tab4 = st.tabs(['Dataset', 'Model', 'Prediction', 'About Me'])
with tab1:
    col1, col2 = st.columns(2)
    with col1:

        st.subheader('About the Dataset')
        st.write('Link: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction')
        st.write('Number of Rows: 303')
        st.write('Number of Columns: 14')
    with col2:

        st.dataframe(df)
    col1,col2 = st.columns(2)
    with col1:
        fig = go.Figure(px.histogram(df, x = 'sex', color = 'sex'))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.density_heatmap(df)
        st.plotly_chart(fig, x = '' ,use_container_width=True)

with tab2:
    st.subheader('Model Info')

with tab3:
    st.subheader('Heart Failure Prediction')
    with st.form('Vitals Input'):
        st.write('Enter the correct details for the given inout fields.')
        input_df = user_input_features()
        submit = st.form_submit_button('Predict')
    if submit:
        st.dataframe(input_df)
        prediction = model.predict()