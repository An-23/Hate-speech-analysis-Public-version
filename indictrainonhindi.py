import torch
from Final.dataloader import train_dataloader_hostile , validation_dataloader_hostile
from Final.train_val import train_epoch, valid_epoch
from tqdm import tqdm
from Final.Modelindic import model, tokenizer

NUM_EPOCHS=30

torch.cuda.empty_cache()
progress =  tqdm(range(1,NUM_EPOCHS+1), desc='Indic training epoch...', leave=True)

for epoch in progress:
    # Train
    train_epoch(train_dataloader_hostile,validation_dataloader_hostile ,model=model, epoch_id=epoch)

    # Validation
    valid_epoch(validation_dataloader_hostile, model=model, epoch_id=epoch)
    
    # Save
    torch.save(model, './Models/Indicbertonhindi.model')
