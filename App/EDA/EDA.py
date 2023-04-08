import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv('https://raw.githubusercontent.com/bumpansy/Heart-Failure-Prediction/main/Model/heart.csv')

def sex_hist():
    fig = go.Figure(px.histogram(df, x = 'sex', color = 'sex'))
    return fig