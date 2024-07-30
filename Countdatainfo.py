from Datas.data_civil import df_train_civil, df_val_civil, df_test_civil
from Datas.data_final import df_train_final, df_val_final, df_test_final
from Datas.data_hasoc import df_train_hasoc, df_val_hasoc, df_test_hasoc
from Datas.data_hostile import df_train_hostile, df_val_hostile, df_test_hostile

print("Train: Civil")
print(df_train_civil.info())
print("Val: Civil")
print(df_val_civil.info())
print("Test: Civil")
print(df_test_civil.info())
print("==============================")


print("Train: Final")
print(df_train_final.info())
print("Val: Final")
print(df_val_final.info())
print("Test: Final")
print(df_test_final.info())
print("==============================")


print("Train: Hasoc")
print(df_train_hasoc.info())
print("Val: Hasoc")
print(df_val_hasoc.info())
print("Test: Hasoc")
print(df_test_hasoc.info())
print("==============================")


print("Train: Hostile")
print(df_train_hostile.info())
print("Val: Hostile")
print(df_val_hostile.info())
print("Test: Hostile")
print(df_test_hostile.info())
print("==============================")



