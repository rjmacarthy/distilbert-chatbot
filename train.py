import torch
import pandas as pd
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
import os
from sklearn.metrics import accuracy_score
from transformers import DistilBertModel, AdamW, get_linear_schedule_with_warmup

from model import DistilBERTClass
from preprocess import preprocess
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DistilBERTClass(torch.nn.Module):
    def __init__(self, num_classes, model):
        super(DistilBERTClass, self).__init__()
        self.num_classes = num_classes
        self.l1 = model
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(device, dtype=torch.long)
        attention_mask = attention_mask.to(device, dtype=torch.long)
        output_1= self.l1(input_ids, attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

def loss_fn(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets)

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    for bi, d in enumerate(data_loader):
        ids = d[0]
        mask = d[1]
        targets = d[2]
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)
        optimizer.zero_grad()
        outputs = model(
            ids,
            mask
        )
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if bi % 10 == 0:
            print(f'bi={bi}, loss={loss}')

def eval_fn(
    data_loader,
    model,
    device
):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, d in enumerate(data_loader):
            ids = d[0]
            mask = d[1]
            targets = d[2]
            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.long)
            outputs = model(
                ids,
                mask
            )
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.argmax(outputs, dim=1).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

def train(train_data):
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    epochs = int(os.environ.get('EPOCHS', 10))

    df = pd.DataFrame(train_data['data'])

    train_dataset = preprocess(df)

    distilbert = DistilBERTClass(len(train_data['data']), model)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    total_steps = len(train_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    for epoch in range(epochs):
        train_fn(train_loader, model, optimizer, device, scheduler)
        outputs, targets = eval_fn(train_loader, distilbert, device)

        print(f'epoch={epoch}, accuracy={accuracy_score(targets, outputs)}')

    save_model(model, 'model.bin')

def save_model(model, path):
    torch.save(model.state_dict(), path)