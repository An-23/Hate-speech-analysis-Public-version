from loguru import logger
import torch
from transformers import DistilBertTokenizer , DistilBertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_distilbert_multilingual_model(nb_labels):
    '''
    Load 'distilbert-base-multilingual-cased' model without any checkpoint
    'distilbert-base-multilingual-cased' for finetuning
    '''
    logger.info(f"transformers.AutoTokenizer : distilbert-base-multilingual")
    logger.info(f"transformers.AutoModel : distilbert-base-multilingual")
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased', num_labels=nb_labels)
    model.to(device)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

    return model, tokenizer

#Load the Distilbert model
model , tokenizer = load_distilbert_multilingual_model(nb_labels=2)
logger.success("Model Distilbert loaded !")