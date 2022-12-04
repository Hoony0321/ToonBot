import model




model = model.ClassifierModel(version="1.0.4", epoch=10, batch_size=32, validation_split=0.2)


# DATA SET 로드
model.load_data()
# model.load_data_version0()

# 모델 설정
model.build_concat_model()
# model.build_model()

# 모델 학습
model.train()

#모델 시각화
model.show_plot()
