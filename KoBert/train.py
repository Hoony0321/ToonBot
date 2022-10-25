from static import MODEL_PATH, MODEL_VERSION
import model
import tensorflow as tf
import loadData
from official.nlp import optimization


# DATA SET 정의
x_train = loadData.getTrainDataSet()
y_train = loadData.getLabelDataSet()

# 손실 함수 정의
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

# 옵티마이저 정의
epochs = 5
steps_per_epoch = int(len(x_train) / 32)
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

# 모델 최적화 함수 & 손실 함수 설정 및 학습
classifier_model = model.build_classifier_model()
classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

history = classifier_model.fit(x_train, y_train, epochs=epochs, verbose=True, validation_split=0.2)

#모델 저장
classifier_model.save(MODEL_PATH + "/model_{}".format(MODEL_VERSION), include_optimizer=False)