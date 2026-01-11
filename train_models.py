import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.makedirs("models", exist_ok=True)

train_df = pd.read_csv("dataset/train.csv")
val_df = pd.read_csv("dataset/val.csv")
test_df = pd.read_csv("dataset/test.csv")

train_texts, train_labels = train_df["text"], train_df["label"]
val_texts, val_labels = val_df["text"], val_df["label"]
test_texts, test_labels = test_df["text"], test_df["label"]

label_encoder = LabelEncoder()
train_y = label_encoder.fit_transform(train_labels)
val_y = label_encoder.transform(val_labels)
test_y = label_encoder.transform(test_labels)

joblib.dump(label_encoder, "models/label_encoder.pkl")

print("\nTraining LinearSVC model...")

tfidf = TfidfVectorizer(max_features=50000)
X_train_tfidf = tfidf.fit_transform(train_texts)
X_val_tfidf = tfidf.transform(val_texts)
X_test_tfidf = tfidf.transform(test_texts)

joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

svm_model = LinearSVC()
svm_model.fit(X_train_tfidf, train_y)

svm_preds = svm_model.predict(X_test_tfidf)
print("\nLinearSVC Accuracy:", accuracy_score(test_y, svm_preds))

joblib.dump(svm_model, "models/linearsvc_model.pkl")

print("\nTraining Logistic Regression...")

log_model = LogisticRegression(max_iter=5000)
log_model.fit(X_train_tfidf, train_y)

log_preds = log_model.predict(X_test_tfidf)
print("\nLogistic Regression Accuracy:", accuracy_score(test_y, log_preds))

joblib.dump(log_model, "models/logistic_model.pkl")

print("\nTraining CNN model...")

tokenizer = Tokenizer(num_words=50000)
tokenizer.fit_on_texts(train_texts)

train_seq = tokenizer.texts_to_sequences(train_texts)
val_seq = tokenizer.texts_to_sequences(val_texts)
test_seq = tokenizer.texts_to_sequences(test_texts)

max_len = 300
train_seq = pad_sequences(train_seq, maxlen=max_len)
val_seq = pad_sequences(val_seq, maxlen=max_len)
test_seq = pad_sequences(test_seq, maxlen=max_len)

joblib.dump(tokenizer, "models/cnn_tokenizer.pkl")

vocab_size = 50000
embedding_dim = 128

cnn_model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    Conv1D(128, 5, activation="relu"),
    GlobalMaxPooling1D(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

cnn_model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

cnn_model.fit(
    train_seq, train_y,
    validation_data=(val_seq, val_y),
    epochs=5,
    batch_size=64
)

cnn_loss, cnn_acc = cnn_model.evaluate(test_seq, test_y)
print("\nCNN Model Accuracy:", cnn_acc)

cnn_model.save("models/cnn_model.h5")

print("\n🎉 Training completed! All three models saved in /models folder.")
