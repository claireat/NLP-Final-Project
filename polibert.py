import torch
from transformers import BertModel, BertTokenizer

# Load pre-trained PoliBERT model and tokenizer
model_name = "nlpaueb/polibert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)