import torchmetrics as tm
import torch
import pandas as pd

# Define the Metrics
accuracy = tm.Accuracy(task='binary', num_labels=2) # Assuming you have 2 labels 
f1score = tm.F1Score(task='binary', num_labels=2) # Add num_labels here as well
recall = tm.Recall(task='binary',num_labels=2)
precision = tm.Precision(task='binary',num_labels=2)
auroc = tm.AUROC(task='binary',num_labels=2)

def metrics(pred_df,target_df,pathmet):
    
    pred= torch.Tensor(pred_df['Hate'].to_numpy())
    target= torch.Tensor((target_df['Hate']>=0.5).astype(int).values).to(int)
    
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'AUROC'],
        'Value': [accuracy(pred, target), f1score(pred, target), precision(pred, target), recall(pred, target), auroc(pred, target)]
    })

    metrics_df['Value'] = metrics_df['Value'].apply(lambda x: x.item())
    metrics_df.to_csv(pathmet,index=None)
    
 
def metricsllm(df,pathmet):
    
    pred= torch.Tensor(df['predicted_label'].to_numpy())
    target= torch.Tensor((df['Hate']>=0.5).astype(int).values).to(int)
    
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'AUROC'],
        'Value': [accuracy(pred, target), f1score(pred, target), precision(pred, target), recall(pred, target), auroc(pred, target)]
    })

    metrics_df['Value'] = metrics_df['Value'].apply(lambda x: x.item())
    metrics_df.to_csv(pathmet,index=None)    
