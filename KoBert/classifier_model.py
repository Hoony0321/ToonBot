# import torch
import tensorflow_text as text
import tensorflow as tf
import tensorflow_hub as hub

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
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  return tf.keras.Model(text_input, net)


text_test = ['this is such an amazing movie!']
classifier_model = build_classifier_model()
bert_raw_result = classifier_model(tf.constant(text_test))
print(tf.sigmoid(bert_raw_result))
