import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

def encoder(train_text):
    for text in train_text:
        yield tokenizer.encode_plus(
            text.replace('-', '').strip().lower(),
            add_special_tokens=True,
            max_length=10,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

def get_pair_data(df):
    data = []
    for _, row in df.iterrows():
        intent = row['intent']
        examples = row['examples'].split('\n')
        for example in examples:
            tuple = (example, intent)
            data.append(tuple)
    return data

def preprocess(df):
    data = get_pair_data(df)

    df = pd.DataFrame(data, columns=['text', 'label'])

    label_encoder = LabelEncoder()

    df['label'] = label_encoder.fit_transform(df['label'])

    train_data, train_labels = df['text'].values, df['label'].values

    train_tokens = list(encoder(train_data))

    ids = np.array([train_tokens[i]['input_ids'] for i in range(len(train_tokens))])
    mask = np.array([train_tokens[i]['attention_mask'] for i in range(len(train_tokens))])
    labels = np.array(train_labels)

    train_dataset = TensorDataset(
        torch.tensor(ids, dtype=torch.long), 
        torch.tensor(mask, dtype=torch.long), 
        torch.tensor(labels, dtype=torch.long)
    )

    return train_dataset