from transformers import BertModel, ertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch import nn, optim

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'


