from kobert_tokenizer import KoBERTTokenizer
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
data = tokenizer.encode("한국어 모델을 공유합니다.")

print(f'토큰화 데이터 : {data}')