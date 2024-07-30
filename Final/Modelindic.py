import torch
from loguru import logger
import transformers
from transformers import AutoTokenizer, AutoModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelClass(torch.nn.Module):
    def __init__(self, tr_model, dropout, target_labels):
        super(ModelClass, self).__init__()
        self.l1 = tr_model
        self.l2 = torch.nn.Dropout(dropout)
        self.l1_dash = tr_model
        self.l2_dash = torch.nn.Dropout(dropout)
        self.hidden_size = 256
        self.lin_final = torch.nn.Linear(self.hidden_size*6, target_labels)


    def forward(self, ids, mask):
        output_1= self.l1(ids, attention_mask = mask)
        output_1_dash= self.l1_dash(ids, attention_mask = mask)
        output_2 = self.l2(output_1[0][:,0,:])
        output_2_dash = self.l2_dash(output_1_dash[0][:,0,:])
        output_4 = torch.squeeze(torch.cat((output_2_dash, output_2), dim=1)).to(device)
        output = self.lin_final(output_4)
        return output


def load_ai4bharat_model(target_labels):
    '''
    Load ai4bharat model without any checkpoint
    ai4bharat for finetuning
    '''
    logger.info(f"transformers.AutoTokenizer : ai4bharat/indic-bert")
    logger.info(f"transformers.AutoModel : ai4bharat/indic-bert")
    tokenizer = transformers.AutoTokenizer.from_pretrained('ai4bharat/indic-bert')
    tr_model = transformers.AutoModel.from_pretrained('ai4bharat/indic-bert')
    tr_model.to(device)
    model = ModelClass(tr_model, 0.3,target_labels)
    model.to(device)
    return model, tokenizer


# Load the model
model, tokenizer = load_ai4bharat_model(target_labels=2)
logger.success("Model Indic loaded !")