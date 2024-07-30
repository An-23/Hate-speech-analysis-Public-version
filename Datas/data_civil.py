import pandas as pd
import numpy as np
from Datas.data_preprocess import preprocess_text


df_train_civil=pd.read_csv('./Dataset/Civil/civil-comments-train.csv')
df_val_civil=pd.read_csv('./Dataset/Civil/civil-comments-val.csv')
df_test_civil=pd.read_csv('./Dataset/Civil/civil-comments-test.csv')


df_train_civil['Text'] = df_train_civil['Text'].apply(preprocess_text)
df_val_civil['Text'] = df_val_civil['Text'].apply(preprocess_text)
df_test_civil['Text'] = df_test_civil['Text'].apply(preprocess_text)

df_train_civil = df_train_civil[~(df_train_civil.Text == ' ')]
df_val_civil = df_val_civil[~(df_val_civil.Text == ' ')]
df_test_civil = df_test_civil[~(df_test_civil.Text == ' ')]


df_train_civil.reset_index(drop=True, inplace=True)
df_val_civil.reset_index(drop=True, inplace=True)
df_test_civil.reset_index(drop=True, inplace=True)


# df_train_civil=df_train_civil[:28]
# df_val_civil=df_val_civil[:12]
# df_test_civil=df_test_civil[:7]