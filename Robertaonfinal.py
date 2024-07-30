import torch
from Final.dataloader import train_dataloader_rob_final , validation_dataloader_rob_final
from Final.train_val import train_epoch, valid_epoch
from tqdm.auto import tqdm
from Final.Modelroberta import model, tokenizer

NUM_EPOCHS=30

torch.cuda.empty_cache()
progress =  tqdm(range(1,NUM_EPOCHS+1), desc='Roberta training epoch...', leave=True)

for epoch in progress:
    # Train
    train_epoch(train_dataloader_rob_final,validation_dataloader_rob_final ,model, epoch_id=epoch)

    # Validation
    valid_epoch(validation_dataloader_rob_final, model, epoch_id=epoch)
    
    # Save
    torch.save(model, './Models/Robertaonfinal.model')
