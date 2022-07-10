from dataclasses import replace
from fastapi import FastAPI
import joblib
import torch
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from textwrap import wrap
from torch import nn
import uvicorn


def remplace(text:str):
    return text.replace("%", " ")


app = FastAPI()

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
# class to model
class BERTSentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(BERTSentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=False)
    self.drop = nn.Dropout(p=0.3)
    self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    _, cls_output = self.bert(
        input_ids = input_ids,
        attention_mask = attention_mask
    )
    drop_output = self.drop(cls_output)
    output = self.linear(drop_output)
    return output


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = BERTSentimentClassifier(2)
model.load_state_dict(torch.load('./models/mod_torch_cpu.pth'))


# model = torch.load('./models/mod_torch.pth', map_location=torch.device('cpu'))


@app.get('/')
def read_root():
    return {"welcome": "Welcome to my api"}

@app.post('/predict/{review_text}')
def classifySentiment(review_text: str):
    review_text = remplace(review_text)
    print("--------------------")
    print("Texto " + str(review_text))
    print("--------------------")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=False)

    encoding_review = tokenizer.encode_plus(
    review_text,
    max_length = 200,
    truncation = True,
    add_special_tokens = True,
    return_token_type_ids = False,
    pad_to_max_length = True,
    return_attention_mask = True,
    return_tensors = 'pt'
    )

    input_ids = encoding_review['input_ids'].to(device)
    attention_mask = encoding_review['attention_mask'].to(device)
    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)
    print("\n".join(wrap(review_text)))
    if prediction:
        return {'Sentimiento': 'Bueno'}
    else:
        return {'Sentimiento': 'Malo'}




