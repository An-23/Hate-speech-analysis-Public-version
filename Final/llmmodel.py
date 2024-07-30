from loguru import logger
import transformers
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer 
from transformers import AutoModelForCausalLM, BitsAndBytesConfig



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


quantization_config = BitsAndBytesConfig(load_in_8bit=True)

def load_openhathi7B_multilingual_model():
    '''
    Load 'LLM' model without any checkpoint for fine-tuning.
    '''
    logger.info("Loading OpenHathi-7B model and tokenizer")
    model = AutoModelForCausalLM.from_pretrained('sarvamai/OpenHathi-7B-Hi-v0.1-Base', torch_dtype=torch.float16, quantization_config=quantization_config, device_map=device)
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained('sarvamai/OpenHathi-7B-Hi-v0.1-Base')
    return model, tokenizer

#Load the LLM model

model , tokenizer = load_openhathi7B_multilingual_model()
logger.success("Model LLM loaded !")

