from Final.dataloader import validation_dataloader_final, validation_dataloader_hasoc, validation_dataloader_civil, validation_dataloader_hostile, validation_dataloader_rob_final
from Evaluation.eval_val import evaluation_val
import torch
import Evaluation.eval_hasoc as eval_hasoc

# Validate for Muril with Final dataset

model1=torch.load('./Models/Murilonfinal.model')
path1='./Predictions/Valpreds/Muril_val_preds.csv'
evaluation_val(model1,path1,validation_dataloader_final)


# Validate for IndicBert with Hostile dataset

model2=torch.load('./Models/Indicbertonhindi.model')
path2='./Predictions/Valpreds/Indicbert_val_preds.csv'
evaluation_val(model2,path2,validation_dataloader_hostile)


# Validate for Roberta with civil comments dataset

model3=torch.load('./Models/Robertaoncivil.model')
path3='./Predictions/Valpreds/RoBerta_val_preds.csv'
evaluation_val(model3,path3,validation_dataloader_civil)


# # Validate for DistilBert with HASOC dataset

model4=torch.load('./Models/Distilbertonhasoc.model')
path4='./Predictions/Valpreds/Distilbert_val_preds.csv'
eval_hasoc.evaluation_val(model4,path4,validation_dataloader_hasoc)


# Validate for ROberta with Final dataset

model5=torch.load('./Models/Robertaonfinal.model')
path5='./Predictions/Valpreds/RoBerta_onfinal_val_preds.csv'
evaluation_val(model5,path5,validation_dataloader_rob_final)

