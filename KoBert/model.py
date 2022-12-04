import tensorflow_text as text
import tensorflow as tf
import tensorflow_hub as hub
from matplotlib import pyplot as plt
from official.nlp import optimization

from KoBert import utils
from KoBert.static import MODEL_PATH

kobert_preprocessor_url = "https://tfhub.dev/jeongukjae/distilkobert_cased_preprocess/1"
kobert_model_url = "https://tfhub.dev/jeongukjae/distilkobert_cased_L-3_H-768_A-12/1"

class ClassifierModel:
  def __init__(self, version, epoch=100, batch_size=32, validation_split=0.2):
    self.version = version
    self.lr = 0.000001
    self.epochs = epoch
    self.batch_size = batch_size
    self.validation_split = validation_split
    self.x_train = None
    self.y_train = None
    self.model = None
    self.history = None

  def save_model(self):
    self.model.save(MODEL_PATH + "/model_{}".format(self.version), include_optimizer=False)

  def load_model(self):
    self.model = tf.saved_model.load(MODEL_PATH + "/model_{}".format(self.version))


  def build_concat_model(self):

    # bert model
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(kobert_preprocessor_url, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(kobert_model_url, trainable=True, name='BERT_encoder')
    text_output = encoder(encoder_inputs)['pooled_output']
    text_output = tf.keras.layers.BatchNormalization()(text_output)
    text_output = tf.keras.layers.Dropout(0.1)(text_output)
    # text_output = tf.keras.layers.Dense(32, activation='relu')(text_output)

    bert_model = tf.keras.Model(inputs=text_input, outputs=text_output)

    # image model
    image_input = tf.keras.layers.Input(shape=(64,64,3), name='image')
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    image_normalized = normalization_layer(image_input)
    image_output = tf.keras.layers.Conv2D(32, 3, activation='relu')(image_normalized)
    image_output = tf.keras.layers.MaxPooling2D()(image_output)
    image_output = tf.keras.layers.Conv2D(32, 3, activation='relu')(image_output)
    image_output = tf.keras.layers.MaxPooling2D()(image_output)
    image_output = tf.keras.layers.Conv2D(32, 3, activation='relu')(image_output)
    image_output = tf.keras.layers.MaxPooling2D()(image_output)
    image_output = tf.keras.layers.Flatten()(image_output)
    image_output = tf.keras.layers.Dense(64, activation='relu')(image_output)
    image_output = tf.keras.layers.BatchNormalization()(image_output)
    image_output = tf.keras.layers.Dropout(0.1)(image_output)


    image_model = tf.keras.Model(inputs=image_input, outputs=image_output)

    # sequential model
    genre_input = tf.keras.layers.Input(shape=(13,), name='genre')

    # concatenate model
    concatenated = tf.keras.layers.concatenate([bert_model.output, genre_input, image_model.output])
    concatenated = tf.keras.layers.Dense(32, activation='relu')(concatenated)
    concatenated = tf.keras.layers.BatchNormalization()(concatenated)
    concat_out = tf.keras.layers.Dense(1, activation=None, name='classifier')(concatenated)

    concat_model = tf.keras.Model([text_input, genre_input, image_input], concat_out)

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
    concat_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    # concat_model.summary()
    self.model = concat_model

  def build_model(self):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(kobert_preprocessor_url, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(kobert_model_url, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)

    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    model = tf.keras.Model(inputs = text_input, outputs=net)

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
    self.model = model

  def load_data(self):
    self.x_train = [utils.load_brief_data(), utils.load_genre_data(), utils.load_image_data()]
    self.y_train = utils.load_label_data()

  def load_data_version0(self):
    self.x_train = utils.load_brief_data()
    self.y_train = utils.load_label_data()


  def train(self):
    history = self.model.fit(self.x_train, self.y_train, epochs=self.epochs, verbose=True, validation_split=self.validation_split)
    self.history = history

  # 수정해야 함.
  # def valid(self):
  #   self.model.predict(self.x_train, split=1.0)

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