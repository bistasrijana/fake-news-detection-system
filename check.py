import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

MAX_WORDS = 50000
MAX_LEN = 300
EMBED_DIM = 128
CHECKPOINT_DIR = "han_checkpoints"
MODELS_DIR = "models"

train_df = pd.read_csv("dataset/train.csv")
texts_train = train_df['text'].astype(str).tolist()

tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(texts_train)

os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(tokenizer, os.path.join(MODELS_DIR, "han_tokenizer.pkl"))
print("Tokenizer rebuilt and saved successfully!")

val_df = pd.read_csv("dataset/val.csv")
label_mapping = {'fake': 0, 'real': 1}
y_val = np.array(val_df['label'].map(label_mapping).tolist())
x_val = pad_sequences(tokenizer.texts_to_sequences(val_df['text']), maxlen=MAX_LEN)

class AttentionLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(shape=(1,), initializer="zeros", trainable=True)
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

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

results = {}
checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".weights.h5")])

for ckpt in checkpoints:
    path = os.path.join(CHECKPOINT_DIR, ckpt)
    epoch = int(ckpt.split("_")[2].split(".")[0])
    model.load_weights(path)
    loss, acc = model.evaluate(x_val, y_val, verbose=0)
    results[epoch] = (acc, loss)
    print(f"Epoch {epoch}: val_acc={acc:.4f}, val_loss={loss:.4f}")

best_epoch = max(results, key=lambda e: results[e][0])
best_acc, best_loss = results[best_epoch]

print("\n---------------------------")
print(" BEST MODEL FOUND ")
print("---------------------------")
print(f"Epoch: {best_epoch}")
print(f"Validation Accuracy: {best_acc:.4f}")
print(f"Validation Loss: {best_loss:.4f}")
print("---------------------------")

best_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"han_step_{best_epoch:03d}.weights.h5")
model.load_weights(best_checkpoint_path)
model.save(os.path.join(MODELS_DIR, "han_model_best.h5"))
print("Best model saved as han_model_best.h5")
