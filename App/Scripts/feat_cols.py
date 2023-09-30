import tensorflow as tf
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/bumpansy/Heart-Failure-Prediction/main/Model/heart.csv')

def make_feat_cols():
    categorical_columns = ['sex','cp','fbs','restecg','exng','slp','caa','thall']
    numeric_columns = ['trtbps','chol','thalachh','oldpeak']
    feat_cols = []
    for x in categorical_columns:
        vocabulary = df[x].unique()
        feat_cols.append(tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(x,vocabulary)))
    for x in numeric_columns:
        feat_cols.append(tf.feature_column.numeric_column(x,dtype = tf.float32))
    age = tf.feature_column.numeric_column("age")
    age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    feat_cols.append(age_buckets)
    return feat_cols

def df_to_dataset(dataframe, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('output')
    return tf.data.Dataset.from_tensor_slices((dict(dataframe), labels)).shuffle(buffer_size = len(dataframe)).batch(batch_size)