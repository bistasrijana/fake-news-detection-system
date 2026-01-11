import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib


MAX_WORDS = 50000
MAX_LEN = 300
EMBED_DIM = 128
BATCH_SIZE = 64
CHECKPOINT_DIR = "han_checkpoints"
MODELS_DIR = "models"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


train_df = pd.read_csv("dataset/train.csv")
val_df = pd.read_csv("dataset/val.csv")

label_mapping = {'fake': 0, 'real': 1}
y_train = np.array(train_df['label'].map(label_mapping).astype(int).tolist())
y_val = np.array(val_df['label'].map(label_mapping).astype(int).tolist())

texts_train = train_df['text'].astype(str).tolist()
texts_val = val_df['text'].astype(str).tolist()


tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(texts_train)

x_train = pad_sequences(tokenizer.texts_to_sequences(texts_train), maxlen=MAX_LEN)
x_val = pad_sequences(tokenizer.texts_to_sequences(texts_val), maxlen=MAX_LEN)


class AttentionLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform",
                                 trainable=True)
        self.b = self.add_weight(shape=(1,),
                                 initializer="zeros",
                                 trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = inputs * a
        return tf.keras.backend.sum(output, axis=1)


input_layer = Input(shape=(MAX_LEN,))
embedding_layer = Embedding(input_dim=MAX_WORDS, output_dim=EMBED_DIM)(input_layer)
lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer)
attention_layer = AttentionLayer()(lstm_layer)
output_layer = Dense(1, activation='sigmoid')(attention_layer)

han_model = Model(inputs=input_layer, outputs=output_layer)
han_model.compile(optimizer=Adam(1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


checkpoint_file = "han_checkpoints/han_step_012.weights.h5"
if os.path.exists(checkpoint_file):
    han_model.load_weights(checkpoint_file)
    last_epoch = 12
    print(f"Resuming training from checkpoint: {checkpoint_file} (epoch {last_epoch})")
else:
    last_epoch = 0
    print("Checkpoint not found. Starting from scratch.")


checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(CHECKPOINT_DIR, "han_step_{epoch:03d}.weights.h5"),
    save_weights_only=True,
    save_freq='epoch',
    verbose=1
)


current_epoch = last_epoch
try:
    while True:
        han_model.fit(
            x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=current_epoch + 1,
            initial_epoch=current_epoch,
            validation_data=(x_val, y_val),
            shuffle=True,
            callbacks=[checkpoint_callback]
        )
        current_epoch += 1
except KeyboardInterrupt:
    print("\nTraining manually stopped. Saving model and tokenizer...")



han_model.save(os.path.join(MODELS_DIR, "han_model.h5"))
joblib.dump(tokenizer, os.path.join(MODELS_DIR, "han_tokenizer.pkl"))

print("HAN training stopped and saved successfully.")
