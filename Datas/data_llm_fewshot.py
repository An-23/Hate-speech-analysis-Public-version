import pandas as pd
from Datas.data_final import df_train_final as train_df

threshold=100

short_texts=train_df[train_df['Text'].str.len()<threshold]

few_shot_non_hate = short_texts[short_texts['Hate'] == 0].sample(n=4, random_state=12)
few_shot_hate = short_texts[short_texts['Hate'] == 1].sample(n=6, random_state=37)

# Combine both samples
few_shot_examples = pd.concat([few_shot_non_hate, few_shot_hate], ignore_index=True)
few_shot_examples=few_shot_examples.sample(frac=1).reset_index(drop=True)

hate_speech_definition = (
    "Hate speech is defined as any speech with or without sny derogatory language either in English or Hindi or Hinglish that attacks a person or group on the basis of attributes such as race, religion, ethnic origin, "
)


# Select a few examples for few-shot learning
few_shot_prompt = hate_speech_definition + "\n\nHere are some examples:\n\n"
for index, row in few_shot_examples.iterrows():
    label = "Yes" if row['Hate'] == 1 else "No"
    example = f"Text: {row['Text']}\nAnswer: {label}\n"
    few_shot_prompt += example + "\n"
