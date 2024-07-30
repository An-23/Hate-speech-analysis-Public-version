import pandas as pd
import numpy as np
from Datas.data_preprocess import preprocess_text


df_train_hasoc=pd.read_csv('./Dataset/HASOC/HASOC.csv')
df_val_hasoc=pd.read_csv('./Dataset/HASOC/HASOC-val.csv')
df_test_hasoc=pd.read_csv('./Dataset/HASOC/HASOC-test.csv')


df_train_hasoc['Text'] = df_train_hasoc['Text'].apply(preprocess_text)
df_val_hasoc['Text'] = df_val_hasoc['Text'].apply(preprocess_text)
df_test_hasoc['Text'] = df_test_hasoc['Text'].apply(preprocess_text)

df_train_hasoc = df_train_hasoc[~(df_train_hasoc.Text == ' ')]
df_val_hasoc = df_val_hasoc[~(df_val_hasoc.Text == ' ')]
df_test_hasoc = df_test_hasoc[~(df_test_hasoc.Text == ' ')]


df_train_hasoc.reset_index(drop=True, inplace=True)
df_val_hasoc.reset_index(drop=True, inplace=True)
df_test_hasoc.reset_index(drop=True, inplace=True)



# df_train_hasoc=df_train_hasoc[:20]
# df_val_hasoc=df_val_hasoc[:9]
# df_test_hasoc=df_test_hasoc[:6]