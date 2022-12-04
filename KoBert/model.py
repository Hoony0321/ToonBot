import tensorflow_text as text
import tensorflow as tf
import tensorflow_hub as hub
from matplotlib import pyplot as plt
from official.nlp import optimization

from KoBert import utils
from KoBert.static import MODEL_PATH

kobert_preprocessor_url = "https://tfhub.dev/jeongukjae/distilkobert_cased_preprocess/1"
kobert_model_url = "https://tfhub.dev/jeongukjae/distilkobert_cased_L-3_H-768_A-12/1"

def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(kobert_preprocessor_url, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(kobert_model_url, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)

  ## 해당 부분에 데이터를 추가하면 될 듯(이미지 벡터화)
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  return tf.keras.Model(text_input, net)

class ClassifierModel:
  def __init__(self, version, epoch=100, batch_size=32):
    self.version = version
    self.lr = 3e-5
    self.epochs = epoch
    self.batch_size = batch_size
    self.x_train = DataSetUtils.loadWebtoonData()
    self.y_train = DataSetUtils.loadLabelData()
    self.model = self._build_model()
    self.history = None

  def _build_model(self):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(kobert_preprocessor_url, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(kobert_model_url, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    model = tf.keras.Model(text_input, net)

    # 손실 함수 정의
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()

    ##옵티마이저 정의
    steps_per_epoch = int(len(self.x_train) / self.batch_size)
    num_train_steps = steps_per_epoch * self.epochs
    num_warmup_steps = int(0.1 * num_train_steps)
    optimizer = optimization.create_optimizer(init_lr=self.lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    # 모델 최적화 함수 & 손실 함수 설정
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

  def train(self, validation_split=0.2):
    history = self.model.fit(self.x_train, self.y_train, epochs=self.epochs, verbose=True, validation_split=validation_split)
    self.history = history

    # 모델 저장
    self.model.save(MODEL_PATH + "/model_{}".format(self.version), include_optimizer=False)
    return history

  def valid(self):
    self.model.predict(self.x_train, split=1.0)

  def show_plot(self):
    history_dict = self.history.history
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



model = ClassifierModel("1.0.3",epoch=50,batch_size=32)
model.train(validation_split=0.2)
model.show_plot()