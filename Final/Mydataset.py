import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, data_df, tokenizer):
        self.data = data_df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        comment = self.data.iloc[index]["Text"]
        label = torch.tensor(self.data.iloc[index]['Hate'].tolist(), dtype=torch.float)

        token_list, attention_mask = self.text_to_token_and_mask(comment)

        return dict(index=index, ids=token_list, mask=attention_mask, labels=label)

    def text_to_token_and_mask(self, input_text):
        tokenization_dict = self.tokenizer.encode_plus(input_text,
                                add_special_tokens=True,
                                max_length=256,
                                padding='max_length',
                                truncation=True,
                                return_attention_mask=True,
                                return_tensors='pt')
        
        token_list = tokenization_dict["input_ids"].flatten()
        attention_mask = tokenization_dict["attention_mask"].flatten()
        return (token_list, attention_mask)
    

