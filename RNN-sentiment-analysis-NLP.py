import keras.preprocessing.text
import numpy as np
import tensorflow as tf
from keras_preprocessing import sequence

vocab_size = 88584
max_len = 250
batch = 64
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(
    num_words=vocab_size
)

print(train_data[0])  # one review
# Making the len of every review 250
train_data = sequence.pad_sequences(train_data, max_len)
test_data = sequence.pad_sequences(test_data, max_len)
# building model with LSTM layer
model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(vocab_size, 32),  # 32 dimensional output embedding
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(
            1, activation="sigmoid"
        ),  # sigmoid for classification into good and bad
    ]
)

model.summary()

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)

results = model.evaluate(test_data, test_labels)
print(results)

word_index = tf.keras.datasets.imdb.get_word_index()


def encode(text):
    tokens = keras.preprocessing.text.text_to_word_sequence(
        text
    )  # splitting the sentence into individual tokens/words
    # replacing words with integers (encoding) if it's in the vocab. otherwise 0
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return sequence.pad_sequences([tokens], max_len)[0]


example = "that movie was just amazing, so wonderful. really loved it and would watch it again because it was great"
example2 = "the movie sucked. I hated it and wouldn't watch it again. was one of the worst things i ever watched. it was disgusting and awful"
encoded = encode(example)
print(encoded)
print(encode(example2))


def predict(text):
    encoded_text = encode(text)
    pred = np.zeros((1, 250))  # because model expects input in form of (x, 250)
    pred[0] = encoded_text
    result = model.predict(pred)
    print(result)


predict(example)
predict(example2)
