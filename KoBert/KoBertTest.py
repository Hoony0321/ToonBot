import torch
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer

text = "한국어 모델을 공유합니다."
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
model = BertModel.from_pretrained('skt/kobert-base-v1')
inputs = tokenizer.batch_encode_plus([text])
out = model(input_ids = torch.tensor(inputs['input_ids']),
              attention_mask = torch.tensor(inputs['attention_mask']))

print(out)