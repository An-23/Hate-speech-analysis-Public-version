import pandas as pd
import numpy as np
from Datas.data_preprocess import preprocess_text


df_train_final=pd.read_csv('./Dataset/Final/final_df_train.csv')
df_val_final=pd.read_csv('./Dataset/Final/final_df_val.csv')
df_test_final=pd.read_csv('./Dataset/Final/final_df_test.csv')


df_train_final['Text'] = df_train_final['Text'].apply(preprocess_text)
df_val_final['Text'] = df_val_final['Text'].apply(preprocess_text)
df_test_final['Text'] = df_test_final['Text'].apply(preprocess_text)

df_train_final = df_train_final[~(df_train_final.Text == ' ')]
df_val_final = df_val_final[~(df_val_final.Text == ' ')]
df_test_final = df_test_final[~(df_test_final.Text == ' ')]


df_train_final.reset_index(drop=True, inplace=True)
df_val_final.reset_index(drop=True, inplace=True)
df_test_final.reset_index(drop=True, inplace=True)


# df_train_final=df_train_final[:35]
# df_val_final=df_val_final[:11]
# df_test_final=df_test_final[:10]