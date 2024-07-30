import torch 
from Datas.data_llm_fewshot import few_shot_prompt
from Final.llmmodel import tokenizer, model
from Datas.data_final import df_test_final
from Datas.data_civil import df_test_civil
from Datas.data_hasoc import df_test_hasoc
from Datas.data_hostile import df_test_hostile
from tqdm.auto import tqdm
import gc
tqdm.pandas()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Setting the max len of texts in examples as 600, any example with more than threshold is removed

threshold =600

df_short_final=df_test_final[df_test_final['Text'].str.len()<threshold]
df_short_final.reset_index(drop=True,inplace=True)


df_short_civil=df_test_civil[df_test_civil['Text'].str.len()<threshold]
df_short_civil.reset_index(drop=True,inplace=True)


df_short_hasoc=df_test_hasoc[df_test_hasoc['Text'].str.len()<threshold]
df_short_hasoc.reset_index(drop=True,inplace=True)


df_short_host=df_test_hostile[df_test_hostile['Text'].str.len()<threshold]
df_short_host.reset_index(drop=True,inplace=True)



BATCH_SIZE =16
        
# Define the batched hate speech detection function
def detect_hate_speech_batch(texts):
    inputs = [few_shot_prompt + f"Text: {text}\nAnswer:" for text in texts]
   
    tokenized_inputs = tokenizer(inputs, return_tensors="pt",max_length=2048, padding=True, truncation=True)
    tokenized_inputs = {key: value.to(device) for key, value in tokenized_inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**tokenized_inputs, max_new_tokens=4, do_sample=True)
    torch.cuda.empty_cache()    

    answers=[]
    for i, output in enumerate(outputs):
        decoded_output = tokenizer.decode(output, skip_special_tokens=True)
        answer=decoded_output.split("Answer: ")[-1].strip()
        answers.append(answer)
        
    return ["yes" in answer.lower() for answer in answers]


def apply_hate_speech_detection(df):
    texts = df['Text'].tolist()
    predictions = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_predictions = detect_hate_speech_batch(batch_texts)
        predictions.extend(batch_predictions)
        
    return predictions


#Applying them on different dataset

df_short_hasoc['predicted_label'] = apply_hate_speech_detection(df_short_hasoc)
df_short_hasoc['predicted_label']=df_short_hasoc['predicted_label'].astype(int)
path1='./Predictions/LLMpreds/hasoc.csv'
df_short_hasoc.sort_index(inplace=True)
df_short_hasoc.to_csv(path1,index=None)



df_short_host['predicted_label'] = apply_hate_speech_detection(df_short_host)
df_short_host['predicted_label']=df_short_host['predicted_label'].astype(int)
path2='./Predictions/LLMpreds/hostile.csv'
df_short_host.sort_index(inplace=True)
df_short_host.to_csv(path2,index=None)



df_short_final['predicted_label'] = apply_hate_speech_detection(df_short_final)
df_short_final['predicted_label']=df_short_final['predicted_label'].astype(int)
path3='./Predictions/LLMpreds/final.csv'
df_short_final.sort_index(inplace=True)
df_short_final.to_csv(path3,index=None)



df_short_civil['predicted_label'] = apply_hate_speech_detection(df_short_civil)
df_short_civil['predicted_label']=df_short_civil['predicted_label'].astype(int)
path4='./Predictions/LLMpreds/civil.csv'
df_short_civil.sort_index(inplace=True)
df_short_civil.to_csv(path4,index=None)

