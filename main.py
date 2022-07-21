import tensorflow as tf
from keras.utils import normalize
from keras.layers import Conv1D, Dense, Dropout, GlobalMaxPooling1D, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
import wandb
from wandb.keras import WandbCallback

wandb.init(project="IMDB sentiment classification")

vocab_size = 10000
vocab_dim = 100
seq_len = 300
num_classes = 2

data = tf.keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = data.load_data(num_words=vocab_size)

x_train = pad_sequences(x_train, value=0, padding="post", maxlen=seq_len)
x_test = pad_sequences(x_test, value=0, padding="post", maxlen=seq_len)

model = tf.keras.models.Sequential(
    [
        Embedding(vocab_size, 16),
        Conv1D(
            filters=256,
            kernel_size=2,
            kernel_initializer="he_normal",
            strides=1,
            padding="VALID",
            activation="relu",
        ),
        GlobalMaxPooling1D(),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["sparse_categorical_accuracy"],
)

history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=20,
    validation_data=(x_test, y_test),
    validation_freq=1,
    callbacks=[WandbCallback()],
)

model.save("model.h5")
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)
