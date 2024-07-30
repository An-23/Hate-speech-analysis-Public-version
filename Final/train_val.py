import numpy as np
from torchmetrics import Metric
import torch 
from loguru import logger
from tqdm.auto import tqdm
import Final.dataloader as dataloader
import Final.Model as Model

LR=2e-5
NUM_EPOCHS = 30

## Metrics for train and validation 

num_classes = 2
train_metric_dict = dict()

from torchmetrics import AUROC, F1Score, MetricCollection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AUROC Macro
auroc_macro = AUROC(task='binary', num_labels=num_classes, average="macro") 
train_metric_dict["auroc_macro"] = auroc_macro

# F1 score global
f1 = F1Score(task='binary', num_labels=num_classes) 
train_metric_dict["f1"] = f1

train_metric = MetricCollection(train_metric_dict)
train_metric.to(device)

validation_metric = train_metric.clone()
validation_metric.to(device)


## Training 

epoch= range(1,NUM_EPOCHS+1)

def train_epoch(train_dataloader,validation_dataloader,model,epoch_id=None):
    model.train()
    logger.info(f"START EPOCH {epoch_id=}")

    loss_list = []
    auroc_macro_list = []

    progress = tqdm(train_dataloader, desc='training batch...', leave=True)
    for batch_id, batch in enumerate(progress):
        if batch_id % 1_000 == 0:
            valid_epoch(validation_dataloader, model, epoch_id=epoch, batch_id=batch_id)

        logger.trace(f"{batch_id=}")
        token_list_batch = batch["ids"].to(device, non_blocking=True)
        attention_mask_batch = batch["mask"].to(device, non_blocking=True)
        label_batch = batch["labels"].to(device, non_blocking=True)

        # Reset gradient
        dataloader.optimizer.zero_grad()

        # Predict
        prediction_batch = model(token_list_batch, attention_mask_batch)
        transformed_prediction_batch = prediction_batch.squeeze()
        transformed_prediction_batch = transformed_prediction_batch[:, 1]

        # Loss
        loss = dataloader.criterion(transformed_prediction_batch.to(torch.float32), label_batch.to(torch.float32))

        # Metrics
        proba_prediction_batch = torch.sigmoid(transformed_prediction_batch)
        train_metrics_collection_dict = train_metric(proba_prediction_batch.to(torch.float32), label_batch.to(torch.int32))
        logger.trace(train_metrics_collection_dict)

        # Backprop
        loss.backward()
        
        # gradient clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        dataloader.optimizer.step()

        # Update progress bar description
        
        auroc_macro_value = float(train_metrics_collection_dict["auroc_macro"])

        loss_list.append(loss.item())
        auroc_macro_list.append(auroc_macro_value)

    loss_mean = np.mean(loss_list)
    logger.info(f"END EPOCH {epoch_id=}")
    logger.info(f"Loss Train Mean : {loss_mean}")
    logger.info(f"AUROC Macro Train Mean : {np.mean(auroc_macro_list)}")



## Validation    

@torch.no_grad()
def valid_epoch(validation_dataloader, model, epoch_id=None, batch_id=None):
    model.eval()
    logger.info(f"START VALIDATION {epoch_id=}{batch_id=}")
    validation_metric.reset()

    loss_list = []
    prediction_list = torch.Tensor([])
    target_list = torch.Tensor([])


    progress = tqdm(validation_dataloader, desc="valid batch...", leave=True)
    for _, batch in enumerate(progress):

        token_list_batch = batch["ids"].to(device,non_blocking=True)
        attention_mask_batch = batch["mask"].to(device,non_blocking=True)
        label_batch = batch["labels"].to(device,non_blocking=True)

        # Predict
        prediction_batch = model(token_list_batch, attention_mask_batch)

        transformed_prediction_batch = prediction_batch.squeeze()
        transformed_prediction_batch = transformed_prediction_batch[:, 1]   

        # Loss
        loss = dataloader.criterion(
            transformed_prediction_batch.to(torch.float32),
            label_batch.to(torch.float32),
        )

        loss_list.append(loss.item())

        proba_prediction_batch = torch.sigmoid(transformed_prediction_batch)
        prediction_list = torch.concat(
            [prediction_list, proba_prediction_batch.cpu()]
        )
        target_list = torch.concat([target_list, label_batch.cpu()])

        # Metrics
        validation_metric(proba_prediction_batch.to(torch.float32), label_batch.to(torch.int32))

    loss_mean = np.mean(loss_list)
    logger.trace(validation_metric.compute())
    logger.info(f"Loss Validation Mean : {loss_mean}")
    logger.info(f"END VALIDATION {epoch_id=}{batch_id=}")
    
    

    

