from loguru import logger
import transformers
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelClassifier(nn.Module):
    def __init__(self, tr_model, nb_labels, dropout_prob=0.25, freeze=False):
        super().__init__()
        self.tr_model = tr_model

        # Stack features of 4 last encoders
        self.hidden_dim = tr_model.config.hidden_size * 4

        # hidden linear for the classification
        self.dropout = nn.Dropout(dropout_prob)
        self.hl = nn.Linear(self.hidden_dim, tr_model.config.hidden_size)

        # Last Linear for the classification
        self.last_l = nn.Linear(tr_model.config.hidden_size, nb_labels)

        # freeze all the parameters if necessary
        for param in self.tr_model.parameters():
            param.requires_grad = not freeze

        # init learning params of last layers
        torch.nn.init.xavier_uniform_(self.hl.weight)
        torch.nn.init.xavier_uniform_(self.last_l.weight)

    def forward(self, ids, mask):

        tr_output = self.tr_model(input_ids=ids,
                                  attention_mask=mask,
                                  output_hidden_states=True)

        # Get all the hidden states
        hidden_states = tr_output['hidden_states']

        # hs_* = [batch_size, padded_seq_len, 768]
        hs_1 = hidden_states[-1][:, 0, :]
        hs_2 = hidden_states[-2][:, 0, :]
        hs_3 = hidden_states[-3][:, 0, :]
        hs_4 = hidden_states[-4][:, 0, :]

        features_vec = torch.cat([hs_1, hs_2, hs_3, hs_4], dim=-1)

        x = self.dropout(features_vec)
        x = self.hl(x)

        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.last_l(x)

        return x
    
    
def load_roberta_model(nb_labels):
    '''
    Load RoBERTa model without any checkpoint
    RoBERTa for finetuning
    '''
    logger.info(f"transformers.AutoTokenizer : xlm-roberta-base")
    logger.info(f"transformers.AutoModel : xlm-roberta-base")
    tokenizer = transformers.AutoTokenizer.from_pretrained('xlm-roberta-base')
    tr_model = transformers.AutoModel.from_pretrained('xlm-roberta-base')
    tr_model.to(device)
    model = ModelClassifier(tr_model, nb_labels, freeze=False)
    model.to(device)
    return model, tokenizer


#Load the roberta model
model , tokenizer = load_roberta_model(nb_labels=2)
logger.success("Model XLM Roberta loaded !")