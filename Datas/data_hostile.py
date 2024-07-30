import pandas as pd
import numpy as np
from Datas.data_preprocess import preprocess_text


df_train_hostile=pd.read_csv('./Dataset/Hostile/Hostile-hindi-train.csv')
df_val_hostile=pd.read_csv('./Dataset/Hostile/Hostile-hindi-val.csv')
df_test_hostile=pd.read_csv('./Dataset/Hostile/Hostile-hindi-test.csv')


df_train_hostile['Text'] = df_train_hostile['Text'].apply(preprocess_text)
df_val_hostile['Text'] = df_val_hostile['Text'].apply(preprocess_text)
df_test_hostile['Text'] = df_test_hostile['Text'].apply(preprocess_text)

df_train_hostile = df_train_hostile[~(df_train_hostile.Text == ' ')]
df_val_hostile = df_val_hostile[~(df_val_hostile.Text == ' ')]
df_test_hostile = df_test_hostile[~(df_test_hostile.Text == ' ')]


df_train_hostile.reset_index(drop=True, inplace=True)
df_val_hostile.reset_index(drop=True, inplace=True)
df_test_hostile.reset_index(drop=True, inplace=True)


# df_train_hostile=df_train_hostile[:18]
# df_val_hostile=df_val_hostile[:8]
# df_test_hostile=df_test_hostile[:9]