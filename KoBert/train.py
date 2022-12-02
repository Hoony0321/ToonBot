from static import MODEL_PATH, MODEL_VERSION
import model
import tensorflow as tf
import DataSetUtils
from official.nlp import optimization
import matplotlib.pyplot as plt



# DATA SET 정의
x_train = DataSetUtils.loadWebtoonData()
y_train = DataSetUtils.loadLabelData()

# train_set, validation_set = DataSetUtils.get_dataset_partitions_tf(dataset, len(dataset), shuffle=True)

# 손실 함수 정의
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

# 옵티마이저 정의
epochs = 100
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

history_dict = history.history
print(history_dict.keys())

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
fig = plt.figure(figsize=(10, 6))



### ====== Training Plot ====== ###

plt.subplot(2, 1, 1)
plt.plot(epochs, loss, 'r', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.tight_layout(h_pad=4)
plt.legend()
plt.show()


### ====== Validation Plot ====== ###

plt.subplot(2, 1, 1)
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.tight_layout(h_pad=4)
plt.legend()
plt.show()


# plt.savefig('loss_{}.png'.format(MODEL_VERSION))
# plt.savefig('accuracy_{}.png'.format(MODEL_VERSION))

#모델 저장
classifier_model.save(MODEL_PATH + "/model_{}".format(MODEL_VERSION), include_optimizer=False)