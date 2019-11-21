from __future__ import absolute_import, division, print_function, unicode_literals

import time
import tensorflow as tf

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


SEQ_LENGTH = 100
CHUNK_LENGTH = SEQ_LENGTH + 1
BATCH_SIZE = 64
BUFFER_SIZE = 10000
EPOCHS = 10


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def main():
    path_to_file = tf.keras.utils.get_file(
        'shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

    vocab = sorted(set(text))
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    text_as_int = np.array([char2idx[c] for c in text])
    # examples_per_epoch = len(text) // CHUNK_LENGTH
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(CHUNK_LENGTH, drop_remainder=True)
    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(
        BATCH_SIZE, drop_remainder=True)

    model = build_model(
        vocab_size=len(vocab),
        embedding_dim=256,
        rnn_units=1024,
        batch_size=BATCH_SIZE)
    model.load_weights(tf.train.latest_checkpoint("./model"))
    model.build(tf.TensorShape([1, None]))

    def generate_text(model, start_string):
        num_generate = 1000
        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)
        text_generated = []
        temperature = 1.0
        model.reset_states()
        for i in range(num_generate):
            predictions = model(input_eval)
            predictions = tf.squeeze(predictions, 0)
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(
                predictions, num_samples=1)[-1, 0].numpy()
            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(idx2char[predicted_id])
        return (start_string + ''.join(text_generated))

    print(generate_text(model, start_string=u"ROMEO: "))


if __name__ == "__main__":
    main()
