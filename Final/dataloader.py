from torch.utils.data import DataLoader
from Final.Mydataset import MyDataset 
import torch
from loguru import logger
import Datas.data_final as data_final
import Datas.data_civil as data_civil
import Datas.data_hasoc as data_hasoc
import Datas.data_hostile as data_hostile



import Final.Model as Model
import Final.Modelindic as Modelindic
import Final.Modelhasoc as Modelhasoc
import Final.Modelroberta as Modelroberta


BATCH_SIZE = 32

#For Final Dataset
train_dataset_final = MyDataset(data_final.df_train_final, Model.tokenizer)
train_dataloader_final = DataLoader(train_dataset_final,
                             batch_size=BATCH_SIZE,
                             shuffle=True)

validation_dataset_final = MyDataset(data_final.df_val_final, Model.tokenizer)
validation_dataloader_final = DataLoader(validation_dataset_final,
                             batch_size=BATCH_SIZE,
                             shuffle=True)



# Final dataset for Roberta Model
train_dataset_rob_final = MyDataset(data_final.df_train_final, Modelroberta.tokenizer)
train_dataloader_rob_final = DataLoader(train_dataset_rob_final,
                             batch_size=BATCH_SIZE,
                             shuffle=True)

validation_dataset_rob_final = MyDataset(data_final.df_val_final, Modelroberta.tokenizer)
validation_dataloader_rob_final = DataLoader(validation_dataset_rob_final,
                             batch_size=BATCH_SIZE,
                             shuffle=True)




#For Civil Dataset
train_dataset_civil = MyDataset(data_civil.df_train_civil, Modelroberta.tokenizer)
train_dataloader_civil = DataLoader(train_dataset_civil,
                             batch_size=BATCH_SIZE,
                             shuffle=True)

validation_dataset_civil = MyDataset(data_civil.df_val_civil, Modelroberta.tokenizer)
validation_dataloader_civil = DataLoader(validation_dataset_civil,
                             batch_size=BATCH_SIZE,
                             shuffle=True)



#For HASOC Dataset
train_dataset_hasoc = MyDataset(data_hasoc.df_train_hasoc, Modelhasoc.tokenizer)
train_dataloader_hasoc = DataLoader(train_dataset_hasoc,
                             batch_size=BATCH_SIZE,
                             shuffle=True)

validation_dataset_hasoc = MyDataset(data_hasoc.df_val_hasoc, Modelhasoc.tokenizer)
validation_dataloader_hasoc = DataLoader(validation_dataset_hasoc,
                             batch_size=BATCH_SIZE,
                             shuffle=True)



#For Hostile Dataset
train_dataset_hostile = MyDataset(data_hostile.df_train_hostile, Modelindic.tokenizer)
train_dataloader_hostile = DataLoader(train_dataset_hostile,
                             batch_size=BATCH_SIZE,
                             shuffle=True)

validation_dataset_hostile = MyDataset(data_hostile.df_val_hostile, Modelindic.tokenizer)
validation_dataloader_hostile = DataLoader(validation_dataset_hostile,
                             batch_size=BATCH_SIZE,
                             shuffle=True)




# Criterions

LR=2e-5

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(Model.model.parameters(), lr=LR, eps=2e-8)
logger.info(optimizer)

