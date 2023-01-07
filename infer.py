from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.preprocessing import LabelEncoder

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
encoder = LabelEncoder()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def infer(
    text,
    model=model,
    tokenizer=tokenizer,
    encoder=encoder,
    device=device
):
    model.load_state_dict(torch.load('model.bin'))
    model.to(device)
    model.eval()
    text = text.replace('-', '').strip().lower()
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=10,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    ids = torch.tensor([inputs['input_ids']], dtype=torch.long).to(device)
    mask = torch.tensor([inputs['attention_mask']], dtype=torch.long).to(device)
    outputs = model(
        ids,
        mask
    )
    return encoder.classes_[torch.argmax(outputs, dim=1).cpu().detach().numpy()[0]]