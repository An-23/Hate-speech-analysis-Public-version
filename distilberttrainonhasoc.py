import torch
#import train_val_hasoc
from Final.dataloader import train_dataloader_hasoc , validation_dataloader_hasoc
from Final.train_val_hasoc import train_epoch, valid_epoch
from tqdm import tqdm
from Final.Modelhasoc import model, tokenizer

NUM_EPOCHS=30

torch.cuda.empty_cache()
progress =  tqdm(range(1,NUM_EPOCHS+1), desc='Bert training epoch...', leave=True)

for epoch in progress:
    # Train
    train_epoch(train_dataloader_hasoc,validation_dataloader_hasoc ,model, epoch_id=epoch)

    # Validation
    valid_epoch(validation_dataloader_hasoc, model, epoch_id=epoch)
    
    # Save
    torch.save(model, './Models/Distilbertonhasoc.model')
