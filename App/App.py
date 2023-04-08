import streamlit as st
import pickle
import pandas as pd
from EDA.EDA import *

st.set_page_config(layout='wide')

url = 'https://raw.githubusercontent.com/bumpansy/Heart-Failure-Prediction/main/Model/heart.csv'
df = pd.read_csv(url,index_col=1)
#model = pickle.load(open('Model\model.pkl', 'rb'))

st.header('Heart Failure Prediction App')
st.write('This app is made to predict the probability of heart failure in a person using his/her vital information.')


tab1,tab2,tab3 = st.tabs(['Dataset', 'Model', 'Prediction'])
with tab1:
    st.subheader('About the Dataset')
    st.write('Link: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction')
    st.write('Number of Rows: 303')
    st.write('Number of Columns: 14')
    st.dataframe(df)
    col1,col2 = st.columns()
    with col1:
        st.plotly_chart(sex_hist)