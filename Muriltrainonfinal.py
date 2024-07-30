import torch
from Final.dataloader import train_dataloader_final , validation_dataloader_final
from Final.train_val import train_epoch, valid_epoch
from tqdm import tqdm
from Final.Model import model, tokenizer

NUM_EPOCHS=30

torch.cuda.empty_cache()
progress =  tqdm(range(1,NUM_EPOCHS+1), desc='Muril training epoch...', leave=True)

for epoch in progress:
    # Train
    train_epoch(train_dataloader_final,validation_dataloader_final ,model=model, epoch_id=epoch)

    # Validation
    valid_epoch(validation_dataloader_final, model=model, epoch_id=epoch)
    
    # Save
    torch.save(model, './Models/Murilonfinal.model')
